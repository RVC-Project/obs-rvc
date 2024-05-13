use std::ops::RangeBounds;

use ndarray::{s, Array1, ArrayViewMut1};

pub trait CopyWithin {
    fn copy_within<R: RangeBounds<usize>>(&mut self, range: R, target: usize);
}

impl<T: Copy> CopyWithin for Array1<T> {
    fn copy_within<R: RangeBounds<usize>>(&mut self, range: R, target: usize) {
        if let Some(slice) = self.as_slice_mut() {
            slice.copy_within(range, target);
            return;
        }

        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => e + 1,
            std::ops::Bound::Excluded(&e) => e,
            std::ops::Bound::Unbounded => self.len(),
        };

        let subrange = self.slice(s![start..end]).to_owned();
        let mut target = self.slice_mut(s![target..target + subrange.len()]);
        target.assign(&subrange);
    }
}

impl<'a, T: Copy> CopyWithin for ArrayViewMut1<'a, T> {
    fn copy_within<R: RangeBounds<usize>>(&mut self, range: R, target: usize) {
        if let Some(slice) = self.as_slice_mut() {
            slice.copy_within(range, target);
            return;
        }

        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => e + 1,
            std::ops::Bound::Excluded(&e) => e,
            std::ops::Bound::Unbounded => self.len(),
        };

        let subrange = self.slice(s![start..end]).to_owned();
        let mut target = self.slice_mut(s![target..target + subrange.len()]);
        target.assign(&subrange);
    }
}
