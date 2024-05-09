use obs_wrapper::{data::FromDataItem, obs_sys::{obs_property_list_add_int, obs_property_list_insert_int, obs_property_t, size_t}, properties::{ComboFormat, ListType}, string::ObsString};

use crate::rvc::enums::{PitchAlgorithm, RvcModelVersion};

macro_rules! enum_to_int_list_type {
    ($t:ty) => {
        impl FromDataItem for $t {
            fn typ() -> obs_wrapper::data::DataType {
                obs_wrapper::data::DataType::Int
            }
            
            unsafe fn from_item_unchecked(item: *mut obs_wrapper::obs_sys::obs_data_item_t) -> Option<Self> {
                let val = obs_wrapper::obs_sys::obs_data_item_get_int(item);

                if <$t>::is_valid(val) {
                    Some(val.into())
                } else {
                    None
                }
            }
            
            unsafe fn set_default_unchecked(obj: *mut obs_wrapper::obs_sys::obs_data_t, name: ObsString, val: Self) {
                obs_wrapper::obs_sys::obs_data_set_int(obj, name.as_ptr(), val.into());
            }
        }

        impl ListType for $t {
            fn format() -> ComboFormat {
                ComboFormat::Int
            }

            fn push_into(self, ptr: *mut obs_property_t, name: ObsString) {
                unsafe {
                    obs_property_list_add_int(ptr, name.as_ptr(), self.into());
                }
            }

            fn insert_into(self, ptr: *mut obs_property_t, name: ObsString, index: usize) {
                unsafe {
                    obs_property_list_insert_int(ptr, index as size_t, name.as_ptr(), self.into());
                }
            }
        }
    };
}

impl From<RvcModelVersion> for i64 {
    fn from(version: RvcModelVersion) -> Self {
        match version {
            RvcModelVersion::V1 => 1,
            RvcModelVersion::V2 => 2,
        }
    }
}

impl From<i64> for RvcModelVersion {
    fn from(val: i64) -> Self {
        match val {
            1 => RvcModelVersion::V1,
            2 => RvcModelVersion::V2,
            _ => RvcModelVersion::V2,
        }
    }
}

impl RvcModelVersion {
    fn is_valid(val: i64) -> bool {
        match val {
            1 | 2 => true,
            _ => false,
        }
    }
}

impl From<PitchAlgorithm> for i64 {
    fn from(algorithm: PitchAlgorithm) -> Self {
        match algorithm {
            PitchAlgorithm::Rmvpe(_) => 1,
        }
    }
}

impl From<i64> for PitchAlgorithm {
    fn from(val: i64) -> Self {
        match val {
            1 => PitchAlgorithm::Rmvpe(None),
            _ => PitchAlgorithm::Rmvpe(None),
        }
    }
}

impl PitchAlgorithm {
    fn is_valid(val: i64) -> bool {
        match val {
            1 => true,
            _ => false,
        }
    }
}

enum_to_int_list_type!(RvcModelVersion);
enum_to_int_list_type!(PitchAlgorithm);
