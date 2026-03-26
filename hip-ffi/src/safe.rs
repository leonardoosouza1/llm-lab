//! Safe Rust wrappers sobre raw HIP FFI.
//! Toda função retorna Result<_, HipError> em vez de hipError_t raw.

use crate::raw::*;
use std::ffi::CStr;
use std::ptr;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum HipError {
    #[error("HIP error {code}: {msg}")]
    Hip { code: u32, msg: String },
}

#[inline]
pub fn hip_check(err: hipError_t) -> Result<(), HipError> {
    if err == hipError_t_hipSuccess {
        return Ok(());
    }
    let msg = unsafe {
        let raw = hipGetErrorString(err);
        if raw.is_null() {
            format!("HIP error {}", err)
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    };
    Err(HipError::Hip { code: err, msg })
}

// ── Device ────────────────────────────────────────────────────────────────────

pub fn hip_set_device(id: i32) -> Result<(), HipError> {
    unsafe { hip_check(hipSetDevice(id)) }
}

pub fn hip_device_sync() -> Result<(), HipError> {
    unsafe { hip_check(hipDeviceSynchronize()) }
}

pub fn hip_mem_info() -> Result<(usize, usize), HipError> {
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe { hip_check(hipMemGetInfo(&mut free, &mut total))? };
    Ok((free, total))
}

// ── Memory ────────────────────────────────────────────────────────────────────

pub unsafe fn hip_malloc<T>(n: usize) -> Result<*mut T, HipError> {
    let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
    hip_check(hipMalloc(&mut ptr, n * std::mem::size_of::<T>()))?;
    Ok(ptr as *mut T)
}

pub unsafe fn hip_free<T>(ptr: *mut T) -> Result<(), HipError> {
    hip_check(hipFree(ptr as *mut std::ffi::c_void))
}

pub unsafe fn hip_h2d<T>(dst: *mut T, src: &[T]) -> Result<(), HipError> {
    hip_check(hipMemcpy(
        dst as *mut std::ffi::c_void,
        src.as_ptr() as *const std::ffi::c_void,
        src.len() * std::mem::size_of::<T>(),
        hipMemcpyKind_hipMemcpyHostToDevice,
    ))
}

pub unsafe fn hip_d2h<T>(dst: &mut [T], src: *const T) -> Result<(), HipError> {
    hip_check(hipMemcpy(
        dst.as_mut_ptr() as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        dst.len() * std::mem::size_of::<T>(),
        hipMemcpyKind_hipMemcpyDeviceToHost,
    ))
}

pub unsafe fn hip_d2d<T>(dst: *mut T, src: *const T, n: usize) -> Result<(), HipError> {
    hip_check(hipMemcpy(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        n * std::mem::size_of::<T>(),
        hipMemcpyKind_hipMemcpyDeviceToDevice,
    ))
}

pub unsafe fn hip_h2d_async<T>(
    dst: *mut T,
    src: &[T],
    stream: hipStream_t,
) -> Result<(), HipError> {
    hip_check(hipMemcpyAsync(
        dst as *mut std::ffi::c_void,
        src.as_ptr() as *const std::ffi::c_void,
        src.len() * std::mem::size_of::<T>(),
        hipMemcpyKind_hipMemcpyHostToDevice,
        stream,
    ))
}

pub unsafe fn hip_d2h_async<T>(
    dst: &mut [T],
    src: *const T,
    stream: hipStream_t,
) -> Result<(), HipError> {
    hip_check(hipMemcpyAsync(
        dst.as_mut_ptr() as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        dst.len() * std::mem::size_of::<T>(),
        hipMemcpyKind_hipMemcpyDeviceToHost,
        stream,
    ))
}

// ── Stream ────────────────────────────────────────────────────────────────────

pub fn hip_stream_create() -> Result<hipStream_t, HipError> {
    let mut s: hipStream_t = ptr::null_mut();
    unsafe { hip_check(hipStreamCreate(&mut s))? };
    Ok(s)
}

pub unsafe fn hip_stream_destroy(s: hipStream_t) -> Result<(), HipError> {
    hip_check(hipStreamDestroy(s))
}

pub unsafe fn hip_stream_sync(s: hipStream_t) -> Result<(), HipError> {
    hip_check(hipStreamSynchronize(s))
}

// ── Module / Kernel ───────────────────────────────────────────────────────────

pub fn hip_module_load(path: &std::path::Path) -> Result<hipModule_t, HipError> {
    use std::ffi::CString;
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut m: hipModule_t = ptr::null_mut();
    unsafe { hip_check(hipModuleLoad(&mut m, c_path.as_ptr()))? };
    Ok(m)
}

pub fn hip_module_load_data(image: &[u8]) -> Result<hipModule_t, HipError> {
    let mut m: hipModule_t = ptr::null_mut();
    unsafe { hip_check(hipModuleLoadData(&mut m, image.as_ptr() as *const _))? };
    Ok(m)
}

pub fn hip_module_get_function(m: hipModule_t, name: &str) -> Result<hipFunction_t, HipError> {
    use std::ffi::CString;
    let c = CString::new(name).unwrap();
    let mut f: hipFunction_t = ptr::null_mut();
    unsafe { hip_check(hipModuleGetFunction(&mut f, m, c.as_ptr()))? };
    Ok(f)
}

/// Lança um kernel com grid e block dims explícitos.
///
/// # Safety
/// - `func` deve ser uma função válida de um módulo carregado
/// - `args` deve conter ponteiros para os argumentos do kernel na ordem correta
pub unsafe fn hip_launch_kernel(
    func: hipFunction_t,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    stream: hipStream_t,
    args: &mut [*mut std::ffi::c_void],
) -> Result<(), HipError> {
    hip_check(hipModuleLaunchKernel(
        func,
        grid.0, grid.1, grid.2,
        block.0, block.1, block.2,
        shared_mem,
        stream,
        args.as_mut_ptr(),
        ptr::null_mut(),
    ))
}

pub unsafe fn hip_module_unload(m: hipModule_t) -> Result<(), HipError> {
    hip_check(hipModuleUnload(m))
}

// ── RAII DeviceBuffer ─────────────────────────────────────────────────────────

/// Buffer tipado na GPU com liberação automática (Drop).
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn alloc(n: usize) -> Result<Self, HipError> {
        let ptr = unsafe { hip_malloc::<T>(n)? };
        Ok(Self { ptr, len: n })
    }

    pub fn from_slice(data: &[T]) -> Result<Self, HipError> {
        let buf = Self::alloc(data.len())?;
        unsafe { hip_h2d(buf.ptr, data)? };
        Ok(buf)
    }

    pub fn to_host(&self) -> Result<Vec<T>, HipError>
    where
        T: Default,
    {
        let mut out = vec![T::default(); self.len];
        unsafe { hip_d2h(&mut out, self.ptr)? };
        Ok(out)
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    /// Ponteiro como argumento de kernel (double-pointer necessário no HIP module API).
    pub fn kernel_arg(&self) -> *mut std::ffi::c_void {
        &self.ptr as *const *mut T as *mut std::ffi::c_void
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hipFree(self.ptr as *mut std::ffi::c_void) };
        }
    }
}
