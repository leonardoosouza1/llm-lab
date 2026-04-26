//! Direct HSA AQL dispatch — bypasses HIP runtime overhead per kernel launch.

use std::ffi::{CStr, CString};
use std::ptr;
use crate::hsa_raw::*;

// HSA constants — bindgen prefixes them with type name, alias for clarity.
const STATUS_SUCCESS: hsa_status_t = hsa_status_t_HSA_STATUS_SUCCESS;
const STATUS_INFO_BREAK: hsa_status_t = hsa_status_t_HSA_STATUS_INFO_BREAK;

// Numeric values from HSA spec (stable; bindgen sometimes drops anonymous enums).
const FENCE_SCOPE_SYSTEM: u16 = 2;
const QUEUE_TYPE_MULTI: u32 = 0;
const DEVICE_TYPE_GPU: u32 = 1;
const PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const PACKET_HEADER_TYPE: u16 = 0;
const PACKET_HEADER_BARRIER: u16 = 8;
const PACKET_HEADER_SCACQUIRE_FENCE_SCOPE: u16 = 9;
const PACKET_HEADER_SCRELEASE_FENCE_SCOPE: u16 = 11;
const KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS: u16 = 0;
const AMD_SEGMENT_GLOBAL: u32 = hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL;
const AMD_MEMORY_POOL_INFO_SEGMENT: hsa_amd_memory_pool_info_t = hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT;
const AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: hsa_amd_memory_pool_info_t = hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS;
const AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1;
const AGENT_INFO_DEVICE: hsa_agent_info_t = hsa_agent_info_t_HSA_AGENT_INFO_DEVICE;
const SIGNAL_CONDITION_LT: hsa_signal_condition_t = hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT;
const WAIT_STATE_ACTIVE: hsa_wait_state_t = hsa_wait_state_t_HSA_WAIT_STATE_ACTIVE;
const PROFILE_FULL: hsa_profile_t = hsa_profile_t_HSA_PROFILE_FULL;
const ROUNDING_DEFAULT: hsa_default_float_rounding_mode_t = hsa_default_float_rounding_mode_t_HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
const SYM_INFO_KERNEL_OBJECT: hsa_executable_symbol_info_t = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT;
const SYM_INFO_KARG_SIZE: hsa_executable_symbol_info_t = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE;
const SYM_INFO_KARG_ALIGN: hsa_executable_symbol_info_t = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT;
const SYM_INFO_GROUP_SIZE: hsa_executable_symbol_info_t = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE;
const SYM_INFO_PRIVATE_SIZE: hsa_executable_symbol_info_t = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE;

#[derive(Debug)]
pub struct HsaError(pub String);
impl std::fmt::Display for HsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl std::error::Error for HsaError {}

fn check(s: hsa_status_t, ctx: &str) -> Result<(), HsaError> {
    if s == STATUS_SUCCESS { return Ok(()); }
    unsafe {
        let mut msg: *const std::os::raw::c_char = ptr::null();
        let _ = hsa_status_string(s, &mut msg);
        let m = if !msg.is_null() {
            CStr::from_ptr(msg).to_string_lossy().into_owned()
        } else {
            format!("hsa_status={}", s)
        };
        Err(HsaError(format!("{}: {}", ctx, m)))
    }
}

pub struct HsaContext {
    pub agent: hsa_agent_t,
    pub queue: *mut hsa_queue_t,
    pub kernarg_pool: hsa_amd_memory_pool_t,
    pub completion_signal: hsa_signal_t,
}

unsafe impl Send for HsaContext {}
unsafe impl Sync for HsaContext {}

#[derive(Clone, Copy, Debug)]
pub struct KernelInfo {
    pub kernel_object: u64,
    pub kernarg_size: u32,
    pub kernarg_align: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
}

pub struct HsaModule {
    pub exec: hsa_executable_t,
    pub reader: hsa_code_object_reader_t,
}

unsafe impl Send for HsaModule {}
unsafe impl Sync for HsaModule {}

struct AgentFindCtx {
    gpu: hsa_agent_t,
    cpu: hsa_agent_t,
}

unsafe extern "C" fn agent_cb(agent: hsa_agent_t, data: *mut std::ffi::c_void) -> hsa_status_t {
    let ctx = &mut *(data as *mut AgentFindCtx);
    let mut device_type: u32 = 0;
    let s = hsa_agent_get_info(agent, AGENT_INFO_DEVICE, &mut device_type as *mut _ as *mut _);
    if s != STATUS_SUCCESS { return STATUS_SUCCESS; }
    if device_type == DEVICE_TYPE_GPU && ctx.gpu.handle == 0 {
        ctx.gpu = agent;
    } else if device_type == 0 /* CPU */ && ctx.cpu.handle == 0 {
        ctx.cpu = agent;
    }
    STATUS_SUCCESS
}

struct PoolFindCtx {
    found: hsa_amd_memory_pool_t,
    got: bool,
}

unsafe extern "C" fn pool_cb(pool: hsa_amd_memory_pool_t, data: *mut std::ffi::c_void) -> hsa_status_t {
    let ctx = &mut *(data as *mut PoolFindCtx);
    let mut segment: u32 = 0;
    let s = hsa_amd_memory_pool_get_info(pool, AMD_MEMORY_POOL_INFO_SEGMENT, &mut segment as *mut _ as *mut _);
    if s != STATUS_SUCCESS { return STATUS_SUCCESS; }
    if segment != AMD_SEGMENT_GLOBAL { return STATUS_SUCCESS; }

    let mut flags: u32 = 0;
    let _ = hsa_amd_memory_pool_get_info(pool, AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &mut flags as *mut _ as *mut _);
    if (flags & AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) == 0 { return STATUS_SUCCESS; }

    ctx.found = pool;
    ctx.got = true;
    STATUS_INFO_BREAK
}

impl HsaContext {
    pub fn init() -> Result<Self, HsaError> {
        unsafe {
            check(hsa_init(), "hsa_init")?;

            let mut afc = AgentFindCtx {
                gpu: hsa_agent_t { handle: 0 },
                cpu: hsa_agent_t { handle: 0 },
            };
            let _ = hsa_iterate_agents(Some(agent_cb), &mut afc as *mut _ as *mut _);
            if afc.gpu.handle == 0 { return Err(HsaError("no GPU agent".into())); }
            if afc.cpu.handle == 0 { return Err(HsaError("no CPU agent".into())); }
            let agent = afc.gpu;

            // KERNARG pool lives on the CPU agent (host-mapped, GPU-accessible).
            let mut kctx = PoolFindCtx {
                found: hsa_amd_memory_pool_t { handle: 0 },
                got: false,
            };
            let _ = hsa_amd_agent_iterate_memory_pools(afc.cpu, Some(pool_cb), &mut kctx as *mut _ as *mut _);
            if !kctx.got { return Err(HsaError("no KERNARG pool on CPU agent".into())); }
            let kernarg_pool = kctx.found;

            let mut queue: *mut hsa_queue_t = ptr::null_mut();
            check(
                hsa_queue_create(
                    agent,
                    1024,
                    QUEUE_TYPE_MULTI,
                    None,
                    ptr::null_mut(),
                    u32::MAX,
                    u32::MAX,
                    &mut queue,
                ),
                "hsa_queue_create",
            )?;

            let mut sig = hsa_signal_t { handle: 0 };
            check(hsa_signal_create(0, 0, ptr::null(), &mut sig), "hsa_signal_create")?;

            Ok(Self { agent, queue, kernarg_pool, completion_signal: sig })
        }
    }

    pub fn alloc_kernarg(&self, size: usize) -> Result<*mut u8, HsaError> {
        let rounded = (size + 15) & !15;
        let rounded = rounded.max(16);
        let mut ptr_out: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check(hsa_amd_memory_pool_allocate(self.kernarg_pool, rounded, 0, &mut ptr_out), "kernarg alloc")?;
            // Make it accessible to the GPU agent.
            let agents = [self.agent];
            let _ = hsa_amd_agents_allow_access(agents.len() as u32, agents.as_ptr(), ptr::null(), ptr_out);
        }
        Ok(ptr_out as *mut u8)
    }

    pub fn free_kernarg(&self, p: *mut u8) {
        unsafe { let _ = hsa_amd_memory_pool_free(p as *mut _); }
    }

    pub fn load_module(&self, hsaco_bytes: &[u8]) -> Result<HsaModule, HsaError> {
        // hipcc --genco wraps the gfx ELF in a Clang Offload Bundle. Strip wrapper.
        let elf_bytes = extract_amdgcn_elf(hsaco_bytes)
            .ok_or_else(|| HsaError("offload bundle: no amdgcn ELF found".into()))?;
        unsafe {
            let mut reader = hsa_code_object_reader_t { handle: 0 };
            check(
                hsa_code_object_reader_create_from_memory(
                    elf_bytes.as_ptr() as *const _,
                    elf_bytes.len(),
                    &mut reader,
                ),
                "code_object_reader",
            )?;

            let mut exec = hsa_executable_t { handle: 0 };
            check(
                hsa_executable_create_alt(PROFILE_FULL, ROUNDING_DEFAULT, ptr::null(), &mut exec),
                "exec_create",
            )?;

            check(
                hsa_executable_load_agent_code_object(exec, self.agent, reader, ptr::null(), ptr::null_mut()),
                "load_code_object",
            )?;
            check(hsa_executable_freeze(exec, ptr::null()), "freeze")?;

            Ok(HsaModule { exec, reader })
        }
    }

    pub fn get_kernel(&self, m: &HsaModule, name: &str) -> Result<KernelInfo, HsaError> {
        let with_kd = format!("{}.kd", name);
        let cn = CString::new(with_kd).unwrap();
        unsafe {
            let mut sym = hsa_executable_symbol_t { handle: 0 };
            let s = hsa_executable_get_symbol_by_name(m.exec, cn.as_ptr(), &self.agent, &mut sym);
            if s != STATUS_SUCCESS {
                let cn2 = CString::new(name).unwrap();
                check(
                    hsa_executable_get_symbol_by_name(m.exec, cn2.as_ptr(), &self.agent, &mut sym),
                    &format!("get_symbol({})", name),
                )?;
            }

            let mut kobj: u64 = 0;
            check(hsa_executable_symbol_get_info(sym, SYM_INFO_KERNEL_OBJECT, &mut kobj as *mut _ as *mut _), "kernel_object")?;
            let mut karg: u32 = 0;
            check(hsa_executable_symbol_get_info(sym, SYM_INFO_KARG_SIZE, &mut karg as *mut _ as *mut _), "kernarg_size")?;
            let mut kalign: u32 = 0;
            check(hsa_executable_symbol_get_info(sym, SYM_INFO_KARG_ALIGN, &mut kalign as *mut _ as *mut _), "kernarg_align")?;
            let mut grp: u32 = 0;
            check(hsa_executable_symbol_get_info(sym, SYM_INFO_GROUP_SIZE, &mut grp as *mut _ as *mut _), "group_size")?;
            let mut prv: u32 = 0;
            check(hsa_executable_symbol_get_info(sym, SYM_INFO_PRIVATE_SIZE, &mut prv as *mut _ as *mut _), "private_size")?;

            Ok(KernelInfo {
                kernel_object: kobj,
                kernarg_size: karg,
                kernarg_align: kalign.max(16),
                group_segment_size: grp,
                private_segment_size: prv,
            })
        }
    }

    /// Async batch dispatch: write packet, doorbell, no wait. Caller must
    /// drain via `sync()` before reading outputs. completion_signal is NOT
    /// modified — assumes caller-managed end-of-batch fence packet or extern sync.
    pub fn dispatch_async(
        &self,
        kernel: &KernelInfo,
        kernarg_addr: *mut u8,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds_bytes: u32,
    ) -> Result<(), HsaError> {
        unsafe {
            let q = &*self.queue;
            let queue_size = q.size as u64;
            let queue_mask = queue_size - 1;
            let idx = hsa_queue_add_write_index_screlease(self.queue, 1);

            while idx - hsa_queue_load_read_index_scacquire(self.queue) >= queue_size {
                std::hint::spin_loop();
            }

            let base = q.base_address as *mut hsa_kernel_dispatch_packet_t;
            let slot = base.add((idx & queue_mask) as usize);

            let mut pkt: hsa_kernel_dispatch_packet_t = std::mem::zeroed();
            pkt.workgroup_size_x = block.0;
            pkt.workgroup_size_y = block.1;
            pkt.workgroup_size_z = block.2;
            pkt.grid_size_x = grid.0;
            pkt.grid_size_y = grid.1;
            pkt.grid_size_z = grid.2;
            pkt.private_segment_size = kernel.private_segment_size;
            pkt.group_segment_size = kernel.group_segment_size + dynamic_lds_bytes;
            pkt.kernel_object = kernel.kernel_object;
            pkt.kernarg_address = kernarg_addr as *mut _;
            // NO completion signal — async path.
            pkt.completion_signal = hsa_signal_t { handle: 0 };

            ptr::copy_nonoverlapping(
                &pkt as *const _ as *const u8,
                slot as *mut u8,
                std::mem::size_of::<hsa_kernel_dispatch_packet_t>(),
            );

            // Header WITH barrier bit: packet must finish before next runs.
            // Required for in-order execution on HSA_QUEUE_TYPE_MULTI; without
            // it kernels may race even if dispatched sequentially.
            let header: u16 = (PACKET_TYPE_KERNEL_DISPATCH << PACKET_HEADER_TYPE)
                | (1u16 << PACKET_HEADER_BARRIER)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
            let setup: u16 = 3u16 << KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
            let full: u32 = (header as u32) | ((setup as u32) << 16);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            (slot as *mut u32).write_volatile(full);

            hsa_signal_store_screlease(q.doorbell_signal, idx as i64);
            Ok(())
        }
    }

    /// Submit a barrier packet that decrements completion_signal when reached.
    /// Call once at end of a batch; then `wait_completion()`.
    pub fn submit_barrier(&self) -> Result<(), HsaError> {
        unsafe {
            let q = &*self.queue;
            let queue_size = q.size as u64;
            let queue_mask = queue_size - 1;
            let idx = hsa_queue_add_write_index_screlease(self.queue, 1);
            while idx - hsa_queue_load_read_index_scacquire(self.queue) >= queue_size {
                std::hint::spin_loop();
            }
            let base = q.base_address as *mut hsa_barrier_and_packet_t;
            let slot = base.add((idx & queue_mask) as usize);

            hsa_signal_store_relaxed(self.completion_signal, 1);

            let mut pkt: hsa_barrier_and_packet_t = std::mem::zeroed();
            pkt.completion_signal = self.completion_signal;
            ptr::copy_nonoverlapping(
                &pkt as *const _ as *const u8,
                slot as *mut u8,
                std::mem::size_of::<hsa_barrier_and_packet_t>(),
            );

            const PACKET_TYPE_BARRIER_AND: u16 = 3;
            let header: u16 = (PACKET_TYPE_BARRIER_AND << PACKET_HEADER_TYPE)
                | (1u16 << PACKET_HEADER_BARRIER)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            (slot as *mut u32).write_volatile(header as u32);
            hsa_signal_store_screlease(q.doorbell_signal, idx as i64);
            Ok(())
        }
    }

    pub fn wait_completion(&self) {
        unsafe {
            let _ = hsa_signal_wait_scacquire(
                self.completion_signal,
                SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                WAIT_STATE_ACTIVE,
            );
        }
    }

    /// Single dispatch via AQL packet, synchronous.
    pub fn dispatch_sync(
        &self,
        kernel: &KernelInfo,
        kernarg_addr: *mut u8,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds_bytes: u32,
    ) -> Result<(), HsaError> {
        unsafe {
            let q = &*self.queue;
            let queue_size = q.size as u64;
            let queue_mask = queue_size - 1;
            let idx = hsa_queue_add_write_index_screlease(self.queue, 1);

            while idx - hsa_queue_load_read_index_scacquire(self.queue) >= queue_size {
                std::hint::spin_loop();
            }

            let base = q.base_address as *mut hsa_kernel_dispatch_packet_t;
            let slot = base.add((idx & queue_mask) as usize);

            hsa_signal_store_relaxed(self.completion_signal, 1);

            // Build packet body. Header set last as atomic release.
            let mut pkt: hsa_kernel_dispatch_packet_t = std::mem::zeroed();
            pkt.workgroup_size_x = block.0;
            pkt.workgroup_size_y = block.1;
            pkt.workgroup_size_z = block.2;
            pkt.grid_size_x = grid.0;
            pkt.grid_size_y = grid.1;
            pkt.grid_size_z = grid.2;
            pkt.private_segment_size = kernel.private_segment_size;
            pkt.group_segment_size = kernel.group_segment_size + dynamic_lds_bytes;
            pkt.kernel_object = kernel.kernel_object;
            pkt.kernarg_address = kernarg_addr as *mut _;
            pkt.completion_signal = self.completion_signal;

            // Write everything except header.
            ptr::copy_nonoverlapping(&pkt as *const _ as *const u8, slot as *mut u8, std::mem::size_of::<hsa_kernel_dispatch_packet_t>());

            // Header + setup as one 32-bit atomic store.
            let header: u16 = (PACKET_TYPE_KERNEL_DISPATCH << PACKET_HEADER_TYPE)
                | (1u16 << PACKET_HEADER_BARRIER)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                | (FENCE_SCOPE_SYSTEM << PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
            let setup: u16 = 3u16 << KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
            let full: u32 = (header as u32) | ((setup as u32) << 16);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            (slot as *mut u32).write_volatile(full);

            hsa_signal_store_screlease(q.doorbell_signal, idx as i64);

            let _ = hsa_signal_wait_scacquire(
                self.completion_signal,
                SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                WAIT_STATE_ACTIVE,
            );

            Ok(())
        }
    }
}

/// Parse Clang Offload Bundle to extract the amdgcn ELF for our GPU target.
/// Bundle layout:
///   "__CLANG_OFFLOAD_BUNDLE__" (24 bytes)
///   u64 num_entries
///   per entry: u64 offset, u64 size, u64 id_len, [u8; id_len] id_string
fn extract_amdgcn_elf(bundle: &[u8]) -> Option<&[u8]> {
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    if bundle.len() < MAGIC.len() + 8 || &bundle[..MAGIC.len()] != MAGIC {
        // Not a bundle — assume raw ELF.
        return Some(bundle);
    }
    let mut p = MAGIC.len();
    let num = u64::from_le_bytes(bundle[p..p + 8].try_into().ok()?) as usize;
    p += 8;
    for _ in 0..num {
        if p + 24 > bundle.len() { return None; }
        let off = u64::from_le_bytes(bundle[p..p + 8].try_into().ok()?) as usize;
        let sz = u64::from_le_bytes(bundle[p + 8..p + 16].try_into().ok()?) as usize;
        let id_len = u64::from_le_bytes(bundle[p + 16..p + 24].try_into().ok()?) as usize;
        p += 24;
        if p + id_len > bundle.len() { return None; }
        let id_str = std::str::from_utf8(&bundle[p..p + id_len]).ok()?;
        p += id_len;
        // Skip "host-" entry. Pick first amdgcn entry.
        if id_str.contains("amdgcn") || id_str.contains("gfx") {
            return Some(&bundle[off..off + sz]);
        }
    }
    None
}

impl Drop for HsaContext {
    fn drop(&mut self) {
        unsafe {
            let _ = hsa_signal_destroy(self.completion_signal);
            if !self.queue.is_null() {
                let _ = hsa_queue_destroy(self.queue);
            }
            let _ = hsa_shut_down();
        }
    }
}

/// Higher-level dispatcher for iara-engine: pre-loads kernels from .hsaco
/// modules and provides a kernarg ring buffer for batched async dispatch.
pub struct HsaDispatcher {
    pub ctx: HsaContext,
    pub modules: Vec<HsaModule>,
    pub kernels: std::collections::HashMap<String, KernelInfo>,
    /// Build-time kernel layouts (offset/size per arg, kernarg_size). Cached
    /// once at startup so dispatch hot path does no HashMap rebuild.
    pub layouts: std::collections::HashMap<&'static str, crate::kernel_layouts::KernelLayout>,
    /// Kernarg ring buffer (host-mapped, GPU-accessible).
    karg_ring: *mut u8,
    karg_ring_size: usize,
    karg_slot_size: usize,
    karg_cursor: std::cell::UnsafeCell<usize>,
}

unsafe impl Send for HsaDispatcher {}
unsafe impl Sync for HsaDispatcher {}

impl HsaDispatcher {
    /// `karg_slot_size` = bytes per dispatch slot (must cover largest kernarg_size, e.g. 512).
    /// `karg_slots` = ring depth (e.g. 1024).
    pub fn new(modules_bytes: &[&[u8]], karg_slot_size: usize, karg_slots: usize) -> Result<Self, HsaError> {
        let ctx = HsaContext::init()?;
        let mut modules = Vec::new();
        let mut kernels = std::collections::HashMap::new();

        // Module-by-module symbol enumeration would be cleaner, but caller already
        // knows the kernel names from llm_kernels.hip. We populate lazily via `register`.
        for bytes in modules_bytes {
            modules.push(ctx.load_module(bytes)?);
        }

        let karg_ring_size = karg_slot_size * karg_slots;
        let karg_ring = ctx.alloc_kernarg(karg_ring_size)?;
        unsafe { std::ptr::write_bytes(karg_ring, 0, karg_ring_size); }

        let layouts = crate::kernel_layouts::all_layouts();
        let mut me = Self {
            ctx,
            modules,
            kernels,
            layouts,
            karg_ring,
            karg_ring_size,
            karg_slot_size,
            karg_cursor: std::cell::UnsafeCell::new(0),
        };
        Ok(me)
    }

    /// Combined info: returns both `KernelInfo` (kernel_object + sizes) and the
    /// build-time kernarg layout for a given kernel. None if either missing.
    pub fn lookup(&self, name: &'static str) -> Option<(&KernelInfo, &crate::kernel_layouts::KernelLayout)> {
        let info = self.kernels.get(name)?;
        let layout = self.layouts.get(name)?;
        Some((info, layout))
    }

    /// Look up a kernel name across loaded modules and cache its KernelInfo.
    pub fn register(&mut self, name: &str) -> Result<KernelInfo, HsaError> {
        if let Some(k) = self.kernels.get(name) {
            return Ok(*k);
        }
        for m in &self.modules {
            if let Ok(info) = self.ctx.get_kernel(m, name) {
                self.kernels.insert(name.to_string(), info);
                return Ok(info);
            }
        }
        Err(HsaError(format!("kernel '{}' not found in any HSA module", name)))
    }

    /// Allocate next slot in kernarg ring. Returns pointer + offset bytes.
    /// Caller writes args into the slot then dispatches.
    /// SAFETY: slot reused after `karg_slots` dispatches — caller must call
    /// `wait_completion()` (or barrier) often enough to avoid reuse-while-running.
    pub fn next_kernarg(&self, kernarg_size: usize) -> *mut u8 {
        unsafe {
            let cursor = &mut *self.karg_cursor.get();
            let slot_addr = self.karg_ring.add(*cursor * self.karg_slot_size);
            *cursor = (*cursor + 1) % (self.karg_ring_size / self.karg_slot_size);
            // Zero-fill before caller writes (hidden args).
            std::ptr::write_bytes(slot_addr, 0, self.karg_slot_size.min(kernarg_size));
            slot_addr
        }
    }

    /// Cached `&KernelInfo`. Kernel must have been `register`ed.
    pub fn get(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.get(name)
    }

    /// Pack explicit arg bytes into kernarg slot AND fill hidden ABI slots.
    /// Caller passes one byte slice per non-hidden ArgSlot in the layout.
    unsafe fn pack_kernarg(
        &self,
        karg: *mut u8,
        layout: &crate::kernel_layouts::KernelLayout,
        arg_bytes: &[&[u8]],
        grid_blocks: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds: u32,
    ) {
        use crate::kernel_layouts::{ArgKind, HiddenKind};
        let mut explicit_idx = 0;
        for slot in layout.args {
            let dst = karg.add(slot.offset as usize);
            match slot.kind {
                ArgKind::Pointer | ArgKind::Scalar => {
                    let bytes = arg_bytes[explicit_idx];
                    explicit_idx += 1;
                    debug_assert!(bytes.len() == slot.size as usize);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                }
                ArgKind::Hidden(hk) => {
                    let val: u64 = match hk {
                        HiddenKind::BlockCountX => grid_blocks.0 as u64,
                        HiddenKind::BlockCountY => grid_blocks.1 as u64,
                        HiddenKind::BlockCountZ => grid_blocks.2 as u64,
                        HiddenKind::GroupSizeX => block.0 as u64,
                        HiddenKind::GroupSizeY => block.1 as u64,
                        HiddenKind::GroupSizeZ => block.2 as u64,
                        HiddenKind::RemainderX => 0,
                        HiddenKind::RemainderY => 0,
                        HiddenKind::RemainderZ => 0,
                        HiddenKind::GridDims => 3,
                        HiddenKind::DynamicLdsSize => dynamic_lds as u64,
                        HiddenKind::Zero => 0,
                    };
                    let src = val.to_le_bytes();
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, slot.size as usize);
                }
            }
        }
    }

    /// Universal sync dispatch with hidden-arg auto-fill.
    pub unsafe fn pack_and_dispatch_sync(
        &self,
        kernel_info: &KernelInfo,
        layout: &crate::kernel_layouts::KernelLayout,
        arg_bytes_per_slot: &[&[u8]],
        grid_blocks: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds: u32,
    ) -> Result<(), HsaError> {
        let karg = self.next_kernarg(kernel_info.kernarg_size as usize);
        self.pack_kernarg(karg, layout, arg_bytes_per_slot, grid_blocks, block, dynamic_lds);
        let grid_wi = (
            grid_blocks.0 * block.0 as u32,
            grid_blocks.1 * block.1 as u32,
            grid_blocks.2 * block.2 as u32,
        );
        // FAST mode: async dispatch (no per-call wait). Engine drains via
        // submit_barrier_and_wait at end of forward.
        if std::env::var("IARA_HSA_FAST").map(|v| v == "1").unwrap_or(false) {
            self.ctx.dispatch_async(kernel_info, karg, grid_wi, block, dynamic_lds)
        } else {
            self.ctx.dispatch_sync(kernel_info, karg, grid_wi, block, dynamic_lds)
        }
    }

    /// Convenience: dispatch with typed args via slices of byte arrays.
    pub unsafe fn dispatch_typed_sync<const N: usize>(
        &self,
        name: &'static str,
        args: [&[u8]; N],
        grid_blocks: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds: u32,
    ) -> Result<(), HsaError> {
        let (kinfo, layout) = self.lookup(name).ok_or_else(|| HsaError(format!("unknown kernel {}", name)))?;
        self.pack_and_dispatch_sync(kinfo, layout, &args, grid_blocks, block, dynamic_lds)
    }

    /// Async variant of dispatch_typed_sync. Caller MUST call
    /// `submit_barrier_and_wait()` (or equivalent) before reading any output
    /// produced by these dispatches. No per-call signal wait — kernel runs
    /// async on the HSA queue. Multiple async dispatches are serialized by
    /// the queue itself (in-order), so no HSA-side race.
    pub unsafe fn dispatch_typed_async<const N: usize>(
        &self,
        name: &'static str,
        args: [&[u8]; N],
        grid_blocks: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds: u32,
    ) -> Result<(), HsaError> {
        let (kinfo, layout) = self.lookup(name).ok_or_else(|| HsaError(format!("unknown kernel {}", name)))?;
        self.pack_and_dispatch_async(kinfo, layout, &args, grid_blocks, block, dynamic_lds)
    }

    /// End-of-batch drain: queue a barrier_and packet that signals
    /// completion_signal, then host waits until the GPU has finished all
    /// prior async dispatches. Replaces N per-call signal waits with one.
    pub fn submit_barrier_and_wait(&self) -> Result<(), HsaError> {
        self.submit_barrier()?;
        self.wait_completion();
        Ok(())
    }

    /// Async variant — caller must call `wait_completion()` (or barrier+wait)
    /// before reading outputs. SAFETY same as `pack_and_dispatch_sync`.
    pub unsafe fn pack_and_dispatch_async(
        &self,
        kernel_info: &KernelInfo,
        layout: &crate::kernel_layouts::KernelLayout,
        arg_bytes_per_slot: &[&[u8]],
        grid_blocks: (u32, u32, u32),
        block: (u16, u16, u16),
        dynamic_lds: u32,
    ) -> Result<(), HsaError> {
        let karg = self.next_kernarg(kernel_info.kernarg_size as usize);
        self.pack_kernarg(karg, layout, arg_bytes_per_slot, grid_blocks, block, dynamic_lds);
        let grid_wi = (
            grid_blocks.0 * block.0 as u32,
            grid_blocks.1 * block.1 as u32,
            grid_blocks.2 * block.2 as u32,
        );
        self.ctx.dispatch_async(kernel_info, karg, grid_wi, block, dynamic_lds)
    }

    /// `grid` is in BLOCKS (HIP convention); converted to workitems internally.
    pub fn dispatch_async(&self, kernel: &KernelInfo, kernarg: *mut u8, grid_blocks: (u32, u32, u32), block: (u16, u16, u16), smem: u32) -> Result<(), HsaError> {
        let grid_workitems = (
            grid_blocks.0 * block.0 as u32,
            grid_blocks.1 * block.1 as u32,
            grid_blocks.2 * block.2 as u32,
        );
        self.ctx.dispatch_async(kernel, kernarg, grid_workitems, block, smem)
    }

    pub fn submit_barrier(&self) -> Result<(), HsaError> { self.ctx.submit_barrier() }
    pub fn wait_completion(&self) { self.ctx.wait_completion(); }
}

impl Drop for HsaDispatcher {
    fn drop(&mut self) {
        if !self.karg_ring.is_null() {
            self.ctx.free_kernarg(self.karg_ring);
        }
    }
}

impl Drop for HsaModule {
    fn drop(&mut self) {
        unsafe {
            let _ = hsa_executable_destroy(self.exec);
            let _ = hsa_code_object_reader_destroy(self.reader);
        }
    }
}
