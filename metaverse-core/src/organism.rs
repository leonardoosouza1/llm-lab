use crate::{OrganismId, Vec2};

#[derive(Debug, Clone)]
pub struct Organism {
    pub id: OrganismId,
    pub pos: Vec2,
    pub complexity: f32,
}

impl Organism {
    pub fn new(id: OrganismId, pos: Vec2, complexity: f32) -> Self {
        Self { id, pos, complexity }
    }
}
