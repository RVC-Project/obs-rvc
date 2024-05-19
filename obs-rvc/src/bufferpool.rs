use core::cell::RefCell;
use core::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

type Used<V> = Arc<RefCell<Vec<V>>>;

const BITS_IN_U32: usize = 32;

fn value_of_index(values: &[u32], index: usize) -> Result<bool, ()> {
    let value_index = index / BITS_IN_U32;
    let offset = index % BITS_IN_U32;

    if let Some(v) = values.get(value_index) {
        if offset < 32 {
            Ok(v & (1 << offset) != 0)
        } else {
            Err(())
        }
    } else {
        Err(())
    }
}

fn update_index(values: &mut [u32], index: usize, value: bool) -> Result<(), ()> {
    let value_index = index / BITS_IN_U32;
    let offset = index % BITS_IN_U32;

    if let Some(v) = values.get_mut(value_index) {
        if offset < 32 {
            let mask = 1 << offset;
            if value {
                *v |= mask;
            } else {
                *v &= !mask;
            }
            Ok(())
        } else {
            Err(())
        }
    } else {
        Err(())
    }
}

/// A "vector of vectors" backed by a single contiguous vector.
/// Allows for mutable borrows of non-overlapping regions.
pub struct BufferPool<V: Default + Clone> {
    buffer: Used<V>,
    buffer_size: usize,
    used: Used<u32>,
}

/// A builder interface for creating a new `BufferPool`.
pub struct BufferPoolBuilder<V: Default + Clone> {
    buffer_size: usize,
    capacity: usize,
    marker: PhantomData<V>,
}

impl<V: Clone + Default> Default for BufferPoolBuilder<V> {
    fn default() -> BufferPoolBuilder<V> {
        BufferPoolBuilder {
            buffer_size: 1024,
            capacity: 0,
            marker: PhantomData {},
        }
    }
}

impl<V: Default + Clone> BufferPoolBuilder<V> {
    pub fn new() -> BufferPoolBuilder<V> {
        BufferPoolBuilder::default()
    }

    /// Set the capacity of the buffer pool - the max number of internal buffers.
    pub fn with_capacity(mut self, capacity: usize) -> BufferPoolBuilder<V> {
        self.capacity = capacity;
        self
    }

    /// Set the buffer size / length of the internal buffers.
    pub fn with_buffer_size(mut self, buffer_size: usize) -> BufferPoolBuilder<V> {
        self.buffer_size = buffer_size;
        self
    }

    pub fn build(self) -> BufferPool<V> {
        BufferPool {
            buffer_size: self.buffer_size,
            buffer: Arc::new(RefCell::new(vec![
                V::default();
                self.capacity * self.buffer_size
            ])),
            used: Arc::new(RefCell::new(vec![
                0;
                if self.capacity == 0 {
                    0
                } else {
                    1 + ((self.capacity - 1) / BITS_IN_U32)
                }
            ])),
        }
    }
}

impl<V: Default + Clone> Default for BufferPool<V> {
    fn default() -> BufferPool<V> {
        BufferPoolBuilder::default().build()
    }
}

impl<V: Default + Clone> BufferPool<V> {
    pub fn builder() -> BufferPoolBuilder<V> {
        BufferPoolBuilder::default()
    }

    fn find_free_index(&self) -> Result<usize, ()> {
        let mut index = 0;
        let max_index = self.capacity();

        loop {
            let used = self.used.borrow();
            let used = used.as_slice();

            if index % BITS_IN_U32 == 0 {
                if let Some(value) = used.get(index / BITS_IN_U32) {
                    if value == &core::u32::MAX {
                        index += BITS_IN_U32;
                        continue;
                    }
                }
            }

            if let Ok(value) = value_of_index(used, index) {
                if !value {
                    return Ok(index);
                } else {
                    index += 1;

                    if max_index <= index {
                        return Err(());
                    }
                }
            } else {
                return Err(());
            }
        }
    }

    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Set all of the values back to their defaults
    pub fn try_clear(&mut self) -> Result<(), ()> {
        if self.is_borrowed() {
            Err(())
        } else {
            let mut buffer = self.buffer.borrow_mut();
            for value in buffer.as_mut_slice().iter_mut() {
                *value = V::default();
            }
            Ok(())
        }
    }

    /// Set all of the values back to their defaults
    ///
    /// # Panics
    /// If any of the buffers have been borrowed.
    pub fn clear(&mut self) {
        if self.try_clear().is_err() {
            panic!("Cannot clear when buffers are borrowed!");
        }
    }

    fn set_index_used(&mut self, index: usize) -> Result<(), ()> {
        let mut used = self.used.borrow_mut();
        let used = used.as_mut_slice();
        update_index(used, index, true)
    }

    fn find_free_index_and_use(&mut self) -> Result<usize, ()> {
        if let Ok(index) = self.find_free_index() {
            self.set_index_used(index).map(|_| index)
        } else {
            Err(())
        }
    }

    /// Return the max number of buffers
    pub fn capacity(&self) -> usize {
        let mut buffer = self.buffer.borrow_mut();
        buffer.as_mut_slice().len() / self.buffer_size
    }

    /// Resize the internal buffers
    ///
    /// # Panics
    /// If any of the buffers have been borrowed.
    pub fn change_buffer_size(&mut self, new_buffer_size: usize) {
        if self.try_change_buffer_size(new_buffer_size).is_err() {
            panic!("Cannot change buffer size when buffers are borrowed!");
        }
    }

    /// Resize the internal buffers
    pub fn try_change_buffer_size(&mut self, new_buffer_size: usize) -> Result<(), ()> {
        let len = self.capacity();
        self.buffer_size = new_buffer_size;
        self.try_resize(len)
    }

    /// Resize both the capacity and buffers
    ///
    /// # Panics
    /// If any of the buffers have been borrowed
    pub fn resize_len_and_buffer(&mut self, new_len: usize, new_buffer_size: usize) {
        self.buffer_size = new_buffer_size;
        self.resize(new_len);
    }

    /// Check whether the buffer pool has no capacity
    pub fn is_empty(&self) -> bool {
        self.capacity() == 0
    }
    
    /// Reserve an additional number of buffers
    /// 
    /// # Panics
    /// If any of the buffers have been borrowed
    pub fn reserve(&mut self, additional: usize) {
        self.resize(self.capacity() + additional);
    }

    /// Reserve an additional number of buffers
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), ()> {
        self.try_resize(self.capacity() + additional)
    }

    /// Checks to see whether any of the internal slices have been borrowed.
    pub fn is_borrowed(&self) -> bool {
        let mut index = 0;
        let max_index = self.capacity();

        let used = self.used.borrow();
        let used = used.as_slice();

        loop {
            if let Ok(value) = value_of_index(used, index) {
                if value {
                    return false;
                } else {
                    index += 1;

                    if max_index <= index {
                        break;
                    }
                }
            } else {
                return false;
            }
        }

        false
    }

    /// Change the number of internal buffers
    ///
    /// # Panics
    /// If any of the internal buffers have been borrowed
    pub fn resize(&mut self, new_len: usize) {
        if self.try_resize(new_len).is_err() {
            panic!("Can't resize when borrowed!");
        }
    }

    /// Change the number of internal buffers
    pub fn try_resize(&mut self, new_len: usize) -> Result<(), ()> {
        if self.is_borrowed() {
            Err(())
        } else {
            let mut buffer = self.buffer.borrow_mut();
            (*buffer).resize_with(new_len * self.buffer_size, V::default);

            let mut used_capacity = self.used.borrow().len() * BITS_IN_U32;

            while used_capacity < new_len {
                let new_len = self.used.borrow().len() + 1;

                self.used.borrow_mut().resize(new_len, 0);

                used_capacity = self.used.borrow().len() * BITS_IN_U32;
            }

            Ok(())
        }
    }

    /// Get a reference to a slice of the `BufferPool` setting the values of the
    /// pool back to their default value.
    pub fn get_cleared_space(&mut self) -> Result<BufferPoolReference<V>, ()> {
        self.get_space().and_then(|mut space| {
            for value in space.as_mut().iter_mut() {
                *value = V::default();
            }

            Ok(space)
        })
    }

    /// Get a reference to a slice of the `BufferPool`.
    pub fn get_space(&mut self) -> Result<BufferPoolReference<V>, ()> {
        self.find_free_index_and_use().and_then(|index| {
            let slice = unsafe {
                (*self.buffer.borrow_mut())
                    .as_mut_ptr()
                    .add(index * self.buffer_size)
            };

            Ok(BufferPoolReference {
                index,
                used: Arc::clone(&self.used),
                parent: Arc::clone(&self.buffer),
                buffer_size: self.buffer_size,
                slice,
            })
        })
    }
}

/// A reference to a slice of the `BufferPool`.
/// When dropped it will finish the borrow and return
/// the space.
pub struct BufferPoolReference<V> {
    index: usize,
    used: Used<u32>,
    // This is only here so it will stay around
    // after the parent is deallocated - never use
    // it!
    #[allow(dead_code)]
    parent: Used<V>,
    slice: *mut V,
    buffer_size: usize,
}

impl<V> AsMut<[V]> for BufferPoolReference<V> {
    fn as_mut(&mut self) -> &mut [V] {
        unsafe { std::slice::from_raw_parts_mut(self.slice, self.buffer_size) }
    }
}

impl<V> AsRef<[V]> for BufferPoolReference<V> {
    fn as_ref(&self) -> &[V] {
        unsafe { std::slice::from_raw_parts(self.slice, self.buffer_size) }
    }
}

impl<V> Drop for BufferPoolReference<V> {
    fn drop(&mut self) {
        let mut used = self.used.borrow_mut();
        let used = used.as_mut_slice();

        if update_index(used, self.index, false).is_err() {
            panic!("Unable to free reference for index {}!", self.index);
        }
    }
}

unsafe impl<V> Send for BufferPoolReference<V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_should_add_capacity() {
        let mut pool: BufferPool<f32> = BufferPool::default();

        assert_eq!(pool.capacity(), 0);

        pool.reserve(1);

        assert_eq!(pool.capacity(), 1);

        pool.reserve(1);

        assert_eq!(pool.capacity(), 2);
    }

    #[test]
    fn it_should_get_space_if_capacity() {
        let mut pool: BufferPool<f32> = BufferPool::default();

        assert_eq!(pool.capacity(), 0);

        assert!(pool.get_space().is_err());

        pool.resize(1);

        let index = pool.get_space().unwrap();

        assert!(pool.get_space().is_err());
        assert_eq!(index.index, 0);
    }

    #[test]
    fn it_should_work_with_interesting_sizes() {
        let sizes: Vec<usize> = vec![12, 100, 1001, 1024, 2048, 4096, 1];

        for buffer_size in sizes.iter() {
            for capacity in sizes.iter() {
                let mut pool: BufferPool<f32> = BufferPoolBuilder::new()
                    .with_buffer_size(*buffer_size)
                    .with_capacity(*capacity)
                    .build();

                assert_eq!(pool.capacity(), *capacity);
                assert_eq!(pool.get_space().is_err(), false);
            }
        }
    }

    #[test]
    fn it_should_return_space_when_deallocated() {
        let mut pool: BufferPool<f32> = BufferPool::default();

        assert_eq!(pool.capacity(), 0);
        pool.reserve(1);

        {
            let index = pool.get_space().unwrap();
            assert!(pool.get_space().is_err());
            assert_eq!(index.index, 0);
        }

        assert!(pool.get_space().is_ok());
    }

    #[test]
    fn it_should_update_internal_buffer() {
        let buffer_size = 10;
        let mut pool: BufferPool<f32> = BufferPool::default();
        pool.change_buffer_size(buffer_size);
        pool.reserve(10);

        let mut a = pool.get_space().unwrap();
        let mut b = pool.get_space().unwrap();

        for value in a.as_mut().iter_mut() {
            *value = 1.;
        }

        for value in b.as_mut().iter_mut() {
            *value = 2.;
        }

        let buffer = pool.buffer.borrow();
        assert_eq!(
            (*buffer)[0..(buffer_size)],
            vec![1. as f32; buffer_size][..]
        );

        assert_eq!(*a.as_ref(), vec![1. as f32; buffer_size][..]);

        let buffer = pool.buffer.borrow();
        assert_eq!(
            (*buffer)[(buffer_size)..(2 * buffer_size)],
            vec![2. as f32; buffer_size][..]
        );

        assert_eq!(*b.as_ref(), vec![2. as f32; buffer_size][..]);
    }

    #[test]
    fn it_should_not_default_space_when_deallocated() {
        let buffer_size = 10;
        let mut pool: BufferPool<f32> = BufferPool::default();
        pool.change_buffer_size(buffer_size);
        pool.reserve(10);

        {
            let mut a = pool.get_space().unwrap();

            for value in a.as_mut().iter_mut() {
                *value = 1.;
            }

            let buffer = pool.buffer.borrow();
            assert_eq!(
                (*buffer)[0..(buffer_size)],
                vec![1. as f32; buffer_size][..]
            );

            assert_eq!(*a.as_ref(), vec![1. as f32; buffer_size][..]);
        }

        let buffer = pool.buffer.borrow();

        assert_eq!(
            (*buffer)[0..(buffer_size)],
            vec![1. as f32; buffer_size][..]
        );
    }

    #[test]
    fn it_should_clear_space_if_explicitly_requested() {
        let buffer_size = 10;
        let mut pool: BufferPool<f32> = BufferPool::default();
        pool.change_buffer_size(buffer_size);
        pool.reserve(10);

        {
            let mut a = pool.get_space().unwrap();

            for value in a.as_mut().iter_mut() {
                *value = 1.;
            }

            let buffer = pool.buffer.borrow();

            assert_eq!(
                (*buffer)[0..(buffer_size)],
                vec![1. as f32; buffer_size][..]
            );

            assert_eq!(*a.as_ref(), vec![1. as f32; buffer_size][..]);
        }

        let space = pool.get_cleared_space().unwrap();

        let buffer = pool.buffer.borrow();

        assert_eq!(
            (*buffer)[0..(buffer_size)],
            vec![0. as f32; buffer_size][..]
        );

        assert_eq!(*space.as_ref(), vec![0. as f32; buffer_size][..]);
    }

    #[test]
    fn it_should_still_work_if_parent_is_dropped() {
        let buffer_size = 10;
        let mut pool: BufferPool<usize> = BufferPool::default();
        pool.change_buffer_size(buffer_size);
        pool.reserve(10);

        let space = pool.get_cleared_space().unwrap();

        drop(pool);

        let value = space.as_ref().iter().fold(0, |a, b| a + b);
        assert_eq!(value, 0);
    }
}
