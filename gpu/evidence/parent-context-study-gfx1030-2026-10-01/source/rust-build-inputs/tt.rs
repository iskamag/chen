//! Fixed-size direct-mapped table. Replacement and layout can be changed independently.
use crate::core::Move;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}
#[derive(Clone, Copy)]
pub struct Entry {
    pub key: u64,
    pub depth: u8,
    pub score: i32,
    pub bound: Bound,
    pub best: Option<Move>,
    pub generation: u8,
}
pub struct TranspositionTable {
    slots: Vec<Option<Entry>>,
    generation: u8,
}
impl TranspositionTable {
    pub fn new(megabytes: usize) -> Self {
        let n =
            (megabytes.clamp(1, 1024) * 1024 * 1024 / std::mem::size_of::<Option<Entry>>()).max(1);
        Self {
            slots: vec![None; n],
            generation: 0,
        }
    }
    pub fn clear(&mut self) {
        self.slots.fill(None);
    }
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
    pub fn get(&self, key: u64) -> Option<Entry> {
        self.slots[key as usize % self.slots.len()].filter(|e| e.key == key)
    }
    pub fn store(&mut self, mut entry: Entry) {
        let index = entry.key as usize % self.slots.len();
        entry.generation = self.generation;
        if self.slots[index].is_none_or(|old| {
            old.key == entry.key || old.generation != self.generation || entry.depth >= old.depth
        }) {
            self.slots[index] = Some(entry);
        }
    }
    pub fn hashfull(&self) -> usize {
        let n = self.slots.len().min(1000);
        self.slots[..n]
            .iter()
            .filter(|e| e.is_some_and(|e| e.generation == self.generation))
            .count()
            * 1000
            / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collisions_replacement_generation_and_clear() {
        let mut table = TranspositionTable::new(1);
        let first = Entry {
            key: 0,
            depth: 5,
            score: 42,
            bound: Bound::Exact,
            best: None,
            generation: 0,
        };
        table.store(first);
        let collision = Entry {
            key: table.slots.len() as u64,
            depth: 2,
            score: -7,
            ..first
        };
        assert!(table.get(collision.key).is_none());
        table.store(collision);
        assert_eq!(table.get(0).unwrap().score, 42);
        table.new_search();
        table.store(collision);
        assert!(table.get(0).is_none());
        assert_eq!(table.get(collision.key).unwrap().score, -7);
        table.clear();
        assert!(table.get(collision.key).is_none());
        assert_eq!(table.hashfull(), 0);
    }
}
