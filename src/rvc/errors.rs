#[derive(Debug)]
pub enum RvcInferError {
    ModelNotLoaded,
    ContentvecNotLoaded,
    Ort(ort::Error),
    NdarrayShapeError(ndarray::ShapeError),
}

impl From<ort::Error> for RvcInferError {
    fn from(err: ort::Error) -> Self {
        RvcInferError::Ort(err)
    }
}

impl From<ndarray::ShapeError> for RvcInferError {
    fn from(err: ndarray::ShapeError) -> Self {
        RvcInferError::NdarrayShapeError(err)
    }
}
