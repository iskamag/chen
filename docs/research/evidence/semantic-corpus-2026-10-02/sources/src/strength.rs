//! Adjustable resource levels. These numbers are not calibrated rating labels.
use crate::SearchLimits;
pub trait StrengthPolicy: Send {
    fn apply(&self, requested: SearchLimits) -> SearchLimits;
}
#[derive(Clone, Copy, Debug)]
pub struct BudgetStrength {
    level: u8,
}
impl BudgetStrength {
    pub fn new(level: u8) -> Result<Self, String> {
        if !(1..=20).contains(&level) {
            return Err("play level must be 1..20".into());
        }
        Ok(Self { level })
    }
    pub fn level(&self) -> u8 {
        self.level
    }
}
impl StrengthPolicy for BudgetStrength {
    fn apply(&self, mut requested: SearchLimits) -> SearchLimits {
        if self.level == 20 {
            return requested;
        }
        let cap = 128_u64 << (self.level - 1);
        requested.nodes = Some(requested.nodes.map_or(cap, |n| n.min(cap)));
        requested.depth = requested.depth.min(1 + (self.level - 1) / 2);
        requested
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn budgets_are_monotonic_and_preserve_stricter_user_limits() {
        let mut previous = 0;
        for level in 1..20 {
            let policy = BudgetStrength::new(level).unwrap();
            let limits = policy.apply(SearchLimits::default());
            assert!(limits.nodes.unwrap() > previous);
            previous = limits.nodes.unwrap();
            assert!(limits.depth <= 10);
            let bounded = policy.apply(SearchLimits {
                depth: 1,
                nodes: Some(8),
                time: Some(Duration::from_millis(10)),
            });
            assert_eq!(bounded.depth, 1);
            assert_eq!(bounded.nodes, Some(8));
            assert_eq!(bounded.time, Some(Duration::from_millis(10)));
        }
        assert!(BudgetStrength::new(0).is_err());
        assert!(BudgetStrength::new(21).is_err());
        assert_eq!(
            BudgetStrength::new(20)
                .unwrap()
                .apply(SearchLimits::default())
                .nodes,
            None
        );
    }
}
