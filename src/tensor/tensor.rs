use rand::RngExt;
use std::cell::{RefCell, RefMut};
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, BitXor, Mul};
use std::rc::Rc;

pub struct TensorInternal {
    _data: Vec<f32>,
    _shape: Vec<usize>,
    _stride: Vec<usize>,

    // Autograd fields:
    _grad: Vec<f32>,
    _gradfn: Option<Box<dyn Fn(&[f32])>>,
    _parents: Vec<Tensor>,
    _requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInternal>>);

impl From<f32> for Tensor {
    fn from(data: f32) -> Self {
        Self(Rc::new(RefCell::new(TensorInternal {
            _data: vec![data],
            _shape: vec![],
            _stride: vec![],
            _grad: vec![],
            _gradfn: None,
            _parents: vec![],
            _requires_grad: false,
        })))
    }
}
impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];

        Self(Rc::new(RefCell::new(TensorInternal {
            _data: data,
            _shape: shape,
            _stride: vec![1],
            _grad: vec![],
            _gradfn: None,
            _parents: vec![],
            _requires_grad: false,
        })))
    }
}
impl From<Vec<Vec<f32>>> for Tensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        if data.is_empty() {
            return Self(Rc::new(RefCell::new(TensorInternal {
                _data: vec![],
                _shape: vec![0, 0],
                _stride: vec![1, 1],
                _grad: vec![],
                _gradfn: None,
                _parents: vec![],
                _requires_grad: false,
            })));
        }

        let cols: usize = data[0].len(); //we know that the first row exists since its past the if
        assert!(
            data.iter().all(|row| row.len() == cols),
            "Dimensions are inconsistent"
        );

        let stride = vec![cols, 1];
        let shape = vec![data.len(), cols];
        let flattened: Vec<f32> = data.into_iter().flatten().collect();
        // stores all the data in row major order( 2d to 1d)

        Self(Rc::new(RefCell::new(TensorInternal {
            _data: flattened,
            _shape: shape,
            _stride: stride,
            _grad: vec![],
            _gradfn: None,
            _parents: vec![],
            _requires_grad: false,
        })))
    }
}

impl Tensor {
    pub(crate) fn set_autograd(&mut self, parents: Vec<Tensor>, gradfn: Box<dyn Fn(&[f32])>) {
        let needs_grad: bool = parents.iter().any(|t| t.0.borrow()._requires_grad);
        if needs_grad {
            let mut internal = self.0.borrow_mut();
            internal._gradfn = Some(gradfn);
            internal._parents = parents;
            internal._requires_grad = true;
        }
    }

    pub fn zero_grad(&self) {
        let mut internal = self.0.borrow_mut();
        if !internal._requires_grad {
            return;
        }

        if internal._grad.len() != internal._data.len() {
            internal._grad = vec![0.0; internal._data.len()];
        } else {
            internal._grad.fill(0.0);
        }
    }

    pub(crate) fn add_grad(&self, incoming_grad: &[f32]) {
        //inplace update of gradient vector
        let mut internal = self.0.borrow_mut();

        if !internal._requires_grad {
            return;
        }
        assert!(
            internal._data.len() == incoming_grad.len(),
            "Gradient and data dimensions do not match"
        );

        if internal._grad.is_empty() {
            internal._grad = incoming_grad.to_vec()
        } else {
            internal
                ._grad
                .iter_mut()
                .zip(incoming_grad.iter())
                .for_each(|(self_val, other_val)| *self_val += *other_val);
        }
    }

    pub fn numel(&self) -> usize {
        self.0.borrow()._data.len()
    }

    fn _buildtopo(
        node: &Tensor,
        visited: &mut HashSet<*const RefCell<TensorInternal>>,
        topo: &mut Vec<Tensor>,
    ) {
        let ptr = Rc::as_ptr(&node.0);
        if visited.contains(&ptr) {
            return;
        } else {
            visited.insert(ptr);
            let parents = node.0.borrow()._parents.clone();
            for parent in parents {
                Self::_buildtopo(&parent, visited, topo)
            }
            topo.push(node.clone());
        }
    }

    pub fn backward(&self) {
        let mut internal = self.0.borrow_mut();
        if internal._grad.is_empty() {
            internal._grad = vec![1.0; internal._data.len()];
        } else {
            internal._grad.fill(1.0);
        }
        drop(internal);
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        Self::_buildtopo(self, &mut visited, &mut topo);
        for node in topo.into_iter().rev() {
            let internal = node.0.borrow();
            if let Some(ref f) = internal._gradfn {
                f(&internal._grad);
            }
        }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();

        assert_eq!(
            data.len(),
            expected_len,
            "Data length ({}) does not match the shape {:?}",
            data.len(),
            shape
        );

        if shape.is_empty() {
            return Self(Rc::new(RefCell::new(TensorInternal {
                _data: data,
                _shape: shape,
                _stride: vec![],
                _grad: vec![],
                _gradfn: None,
                _parents: vec![],
                _requires_grad: false,
            })));
        }

        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        Self(Rc::new(RefCell::new(TensorInternal {
            _data: data,
            _shape: shape,
            _stride: stride,
            _grad: vec![],
            _gradfn: None,
            _parents: vec![],
            _requires_grad: false,
        })))
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();

        Self::new(data, shape)
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = vec![0.0; size];
        Self::new(data, shape)
    }

    pub fn kaiming(shape: Vec<usize>) -> Self {
        //will be used by default to create a linear layer
        assert!(
            shape.len() == 2,
            "shape must be 2-dimensional for kaiming initialization"
        );
        let in_dim: usize = shape[shape.len() - 2];
        let size: usize = shape.iter().product();
        let bound: f32 = (6.0 / in_dim as f32).sqrt();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random_range(-bound..bound)).collect();

        Self::new(data, shape)
    }

    pub fn xavier(shape: Vec<usize>) -> Self {
        
        assert!(
            shape.len() == 2,
            "shape must be 2-dimensional for xavier initialization"
        );
        let in_dim: usize = shape[shape.len() - 2];
        let out_dim: usize = shape[shape.len() - 1];
        let size: usize = shape.iter().product();
        let bound: f32 = (6.0 / (out_dim + in_dim) as f32).sqrt();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| rng.random_range(-bound..bound)).collect();

        Self::new(data, shape)
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow()._requires_grad
    }

    pub fn with_requires_grad(self) -> Self {
        self.0.borrow_mut()._requires_grad = true;
        self
    }

    pub fn set_requires_grad(&mut self, boolean_value: bool) {
        self.0.borrow_mut()._requires_grad = boolean_value;
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow()._shape.clone()
    }

    pub fn stride(&self) -> Vec<usize> {
        self.0.borrow()._stride.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.borrow()._data.clone()
    }

    pub fn set_data(&self, new_data: Vec<f32>) {
        self.0.borrow_mut()._data = new_data;
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        let internal = self.0.borrow();
        if internal._requires_grad && !internal._grad.is_empty() {
            Some(internal._grad.clone())
        } else {
            None
        }
    }

    pub fn t(&self) -> Self {
        let s = self.0.borrow();
        let n = s._shape.len();
        assert!(
            n >= 2,
            "transpose requires at least 2 dimensions, got {:?}",
            s._shape
        );

        let mut new_shape = s._shape.clone();
        new_shape.swap(n - 2, n - 1);

        let rows = s._shape[n - 2];
        let cols = s._shape[n - 1];
        let mat_size = rows * cols;

        let mut new_data = vec![0.0; s._data.len()];
        const BLOCK_SIZE: usize = 32; //L1 cache optimizations

        for (src_mat, dst_mat) in s
            ._data
            .chunks_exact(mat_size)
            .zip(new_data.chunks_exact_mut(mat_size))
        {
            for r_block in (0..rows).step_by(BLOCK_SIZE) {
                for c_block in (0..cols).step_by(BLOCK_SIZE) {
                    let r_end = (r_block + BLOCK_SIZE).min(rows);
                    let c_end = (c_block + BLOCK_SIZE).min(cols);

                    for r in r_block..r_end {
                        let r_offset = r * cols;
                        for c in c_block..c_end {
                            dst_mat[c * rows + r] = src_mat[r_offset + c];
                        }
                    }
                }
            }
        }

        drop(s);
        let mut output = Tensor::new(new_data, new_shape.clone());
        let s_clone = self.clone();
        output.set_autograd(
            vec![self.clone()],
            Box::new(move |grad| {
                let grad_Tensor = Tensor::new(grad.to_vec(), new_shape.clone()).t();
                s_clone.add_grad(&grad_Tensor.0.borrow()._data);
            }),
        );
        output
    }

    pub fn item(&self) -> f32 {
        let internal = self.0.borrow();
        assert!(
            internal._data.len() == 1,
            "Expected 1 element, got {}",
            internal._data.len()
        );
        internal._data[0]
    }
    pub fn item_mut(&self) -> RefMut<'_, f32> {
        let internal = self.0.borrow_mut();
        assert!(
            internal._data.len() == 1,
            "Expected 1 element, got {}",
            internal._data.len()
        );
        RefMut::map(internal, |i| &mut i._data[0])
    }
    pub fn at(&self, indices: &[usize]) -> f32 {
        let internal = self.0.borrow();
        assert!(
            indices.len() == internal._shape.len(),
            "Expected {} indices, got {}",
            internal._shape.len(),
            indices.len()
        );
        for (&idx, &dim) in indices.iter().zip(internal._shape.iter()) {
            assert!(idx < dim, "Index out of bounds");
        }

        let flat_index: usize = indices
            .iter()
            .zip(&internal._stride)
            .map(|(idx, stride)| idx * stride)
            .sum();

        internal._data[flat_index]
    }
    pub fn at_mut(&self, indices: &[usize]) -> RefMut<'_, f32> {
        let internal = self.0.borrow_mut();
        assert!(
            indices.len() == internal._shape.len(),
            "Expected {} indices, got {}",
            internal._shape.len(),
            indices.len()
        );
        for (&idx, &dim) in indices.iter().zip(internal._shape.iter()) {
            assert!(idx < dim, "Index out of bounds");
        }

        let flat_index: usize = indices
            .iter()
            .zip(&internal._stride)
            .map(|(idx, stride)| idx * stride)
            .sum();

        RefMut::map(internal, move |i| &mut i._data[flat_index])
    }

    fn fmt_recurse(&self, f: &mut fmt::Formatter<'_>, dim: usize, offset: usize) -> fmt::Result {
        let internal = self.0.borrow();
        //base case
        if internal._shape.is_empty() {
            return write!(f, "{}", internal._data[0]); //since the shape is empty, there is no shape its just a point
        }

        if internal._shape.len() - 1 == dim {
            //we are at the inner most level of the tensor
            write!(f, "[")?;
            for i in 0..internal._shape[dim] {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", internal._data[offset + i * internal._stride[dim]])?;
            }
            return write!(f, "]");
        }
        write!(f, "[")?;
        for i in 0..internal._shape[dim] {
            if i > 0 {
                write!(f, ",")?;

                if internal._shape.len() - 1 - dim > 1 {
                    write!(f, "\n")?;
                }
                write!(f, "\n")?;
                for _ in 0..=dim {
                    write!(f, " ")?;
                }
            }
            let next_off = offset + i * internal._stride[dim];
            self.fmt_recurse(f, dim + 1, next_off)?;
        }
        return write!(f, "]");
    }
}
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let internal = self.0.borrow();
        if internal._data.is_empty() {
            return write!(f, "[]");
        }
        drop(internal);
        self.fmt_recurse(f, 0, 0)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        let s = self.0.borrow();
        let o = other.0.borrow();
        // 1. Identical shapes: standard element-wise addition
        if s._shape == o._shape {
            let new_data: Vec<f32> = s._data.iter().zip(&o._data).map(|(a, b)| a + b).collect();

            let shape = s._shape.clone();
            drop(s);
            drop(o);
            let mut output = Tensor::new(new_data, shape);
            //use loading helper
            let s_clone = self.clone();
            let o_clone = other.clone();
            output.set_autograd(
                vec![self.clone(), other.clone()], //parents
                Box::new(move |grad| {
                    //gradfn
                    s_clone.add_grad(grad);
                    o_clone.add_grad(grad);
                }),
            );

            return output;
        }
        //this means other is the smaller (bias) and self is the batches tensor that needs bias added to each level
        if s._shape.len() > o._shape.len() && s._shape.ends_with(&o._shape) {
            let newdata: Vec<f32> = s
                ._data
                .iter()
                .zip(o._data.iter().cycle())
                .map(|(a, b)| a + b)
                .collect();
            let shape = s._shape.clone();
            let o_len = o._data.len();
            drop(s);
            drop(o);
            let mut output = Tensor::new(newdata, shape);
            let s_clone = self.clone();
            let o_clone = other.clone();
            output.set_autograd(
                vec![self.clone(), other.clone()], //parents
                Box::new(move |grad| {
                    //gradfn
                    s_clone.add_grad(grad);

                    if o_clone.requires_grad() {
                        let mut o_grad = vec![0.0; o_len];
                        for (i, &g) in grad.iter().enumerate() {
                            o_grad[i % o_len] += g;
                        }
                        o_clone.add_grad(&o_grad);
                    }
                }),
            );
            return output;
        } else if o._shape.len() > s._shape.len() && o._shape.ends_with(&s._shape) {
            drop(s);
            drop(o);
            return other + self; //other way around ez
        } else {
            panic!(
                "Error! Can not add tensors of different shapes: {:?} vs {:?}",
                s._shape, o._shape
            );
        }
    }
}
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        &self + &other
    }
}
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        &self + other
    }
}
impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        self + &other
    }
}
impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, scalar: f32) -> Self::Output {
        let s = self.0.borrow();
        let new_data: Vec<f32> = s._data.iter().map(|a| a + scalar).collect();
        let shape = s._shape.clone();
        drop(s);
        let mut output = Tensor::new(new_data, shape);
        let s_clone = self.clone();
        output.set_autograd(
            vec![self.clone()],
            Box::new(move |grad| {
                s_clone.add_grad(grad);
            }),
        );
        output
    }
}
impl Add<&Tensor> for f32 {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        other + self
    }
}
impl Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self, scalar: f32) -> Self::Output {
        &self + scalar
    }
}
impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        &other + self
    }
}

//element wise multiplication
impl BitXor<&Tensor> for &Tensor {
    type Output = Tensor;
    fn bitxor(self, other: &Tensor) -> Tensor {
        let s = self.0.borrow();
        let o = other.0.borrow();
        assert_eq!(
            s._shape, o._shape,
            "Error! Can not perform element-wise multiplication on tensors of different shapes: {:?} vs {:?}",
            s._shape, o._shape
        );

        let new_data: Vec<f32> = s._data.iter().zip(&o._data).map(|(a, b)| a * b).collect();
        let shape = s._shape.clone();
        drop(s);
        drop(o);

        let mut output = Tensor::new(new_data, shape);
        let s_clone = self.clone();
        let o_clone = other.clone();
        output.set_autograd(
            vec![self.clone(), other.clone()],
            Box::new(move |grad| {
                let s_req = s_clone.requires_grad();
                let o_req = o_clone.requires_grad();

                if s_req && o_req {
                    // Both need grad: Single loop pass!
                    let s = s_clone.0.borrow();
                    let o = o_clone.0.borrow();

                    let mut s_grad = Vec::with_capacity(grad.len());
                    let mut o_grad = Vec::with_capacity(grad.len());

                    for i in 0..grad.len() {
                        let g = grad[i];
                        s_grad.push(g * o._data[i]);
                        o_grad.push(g * s._data[i]);
                    }

                    // Drop borrows before calling add_grad
                    drop(s);
                    drop(o);

                    s_clone.add_grad(&s_grad);
                    o_clone.add_grad(&o_grad);
                } else if s_req {
                    // Only self needs grad: only borrow `other`
                    let o = o_clone.0.borrow();
                    let s_grad: Vec<f32> = grad
                        .iter()
                        .zip(o._data.iter())
                        .map(|(&g, &other_val)| g * other_val)
                        .collect();
                    drop(o);
                    s_clone.add_grad(&s_grad);
                } else if o_req {
                    // Only other needs grad: only borrow `self`
                    let s = s_clone.0.borrow();
                    let o_grad: Vec<f32> = grad
                        .iter()
                        .zip(s._data.iter())
                        .map(|(&g, &self_val)| g * self_val)
                        .collect();
                    drop(s);
                    o_clone.add_grad(&o_grad);
                }
            }),
        );

        output
    }
}
impl BitXor<Tensor> for &Tensor {
    type Output = Tensor;
    fn bitxor(self, other: Tensor) -> Tensor {
        self ^ &other
    }
}
impl BitXor<&Tensor> for Tensor {
    type Output = Tensor;
    fn bitxor(self, other: &Tensor) -> Tensor {
        &self ^ other
    }
}
impl BitXor<Tensor> for Tensor {
    type Output = Tensor;
    fn bitxor(self, other: Tensor) -> Tensor {
        &self ^ &other
    }
}

//matmul
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor; //TODO gradfn vvv
    fn mul(self, other: &Tensor) -> Tensor {
        let s = self.0.borrow();
        let o = other.0.borrow();
        if s._shape.is_empty() || o._shape.is_empty() {
            panic!(
                "Error! Can not perform matmul on 0D tensors: {:?} vs {:?}",
                s._shape, o._shape
            );
        }
        if s._shape.len() == 1 && o._shape.len() == 1 {
            assert!(
                s._shape == o._shape,
                "Error! Can not perform matmul on tensors of different shapes: {:?} vs {:?}",
                s._shape,
                o._shape
            );
            let dotproduct: f32 = s._data.iter().zip(&o._data).map(|(a, b)| a * b).sum();
            drop(s);
            drop(o);
            let mut output = Tensor::new(vec![dotproduct], vec![1]);
            let s_clone = self.clone();
            let o_clone = other.clone();
            output.set_autograd(
                vec![self.clone(), other.clone()],
                Box::new(move |grad| {
                    let s_req = s_clone.requires_grad();
                    let o_req = o_clone.requires_grad();

                    if s_req && o_req {
                        //Both need gradient updates
                        let s = s_clone.0.borrow();
                        let o = o_clone.0.borrow();

                        let mut s_grad = Vec::with_capacity(grad.len());
                        let mut o_grad = Vec::with_capacity(grad.len());

                        let g = grad[0]; //since 1dx1d is a scalar, grad is 1 element
                        for i in 0..s._data.len() {
                            s_grad.push(g * o._data[i]);
                            o_grad.push(g * s._data[i]);
                        }
                        drop(s);
                        drop(o);

                        s_clone.add_grad(&s_grad);
                        o_clone.add_grad(&o_grad);
                    } else if s_req {
                        let s = s_clone.0.borrow();
                        let o = o_clone.0.borrow();

                        let mut s_grad = Vec::with_capacity(grad.len());
                        let g = grad[0]; //since 1dx1d is a scalar, grad is 1 element
                        for i in 0..s._data.len() {
                            s_grad.push(g * o._data[i]);
                        }
                        drop(s);
                        drop(o);
                        s_clone.add_grad(&s_grad);
                    } else if o_req {
                        let s = s_clone.0.borrow();
                        let o = o_clone.0.borrow();
                        let mut o_grad = Vec::with_capacity(grad.len());
                        let g = grad[0]; //since 1dx1d is a scalar, grad is 1 element
                        for i in 0..grad.len() {
                            o_grad.push(g * s._data[i]);
                        }
                        drop(o);
                        drop(s);
                        o_clone.add_grad(&o_grad);
                    }
                }),
            );
            return output;
        }
        //calculating the output shape (and validity check)
        let mut a_shape = s._shape.clone();
        let mut b_shape = o._shape.clone();
        let mut a_is_1D: bool = false;
        let mut b_is_1D: bool = false;

        if s._shape.len() == 1 {
            a_is_1D = true;
            a_shape.insert(0, 1);
        }
        if o._shape.len() == 1 {
            b_is_1D = true;
            b_shape.push(1);
        }

        let self_cols = a_shape[a_shape.len() - 1];
        let self_rows = a_shape[a_shape.len() - 2];
        let other_rows = b_shape[b_shape.len() - 2];
        let other_cols = b_shape[b_shape.len() - 1];
        assert!(
            self_cols == other_rows,
            "Error! Can not perform matmul on tensors of different shapes: {:?} vs {:?}",
            a_shape,
            b_shape
        );

        let mut outbatch = vec![];

        let mut batch_a = a_shape[0..a_shape.len() - 2].to_vec();
        batch_a.reverse();
        let mut batch_b = b_shape[0..b_shape.len() - 2].to_vec();
        batch_b.reverse();

        for i in 0..batch_a.len().max(batch_b.len()) {
            let a_i = if i < batch_a.len() { batch_a[i] } else { 1 };
            let b_i = if i < batch_b.len() { batch_b[i] } else { 1 };

            if a_i == b_i || a_i == 1 || b_i == 1 {
                outbatch.push(a_i.max(b_i));
            } else {
                panic!(
                    "Error! Can not perform matmul on tensors of different shapes: {:?} vs {:?}",
                    a_shape, b_shape
                );
            }
        }

        //reconstructing output shape
        outbatch.reverse();
        let numbatches = outbatch.iter().product();
        if !a_is_1D {
            outbatch.push(self_rows);
        }
        if !b_is_1D {
            outbatch.push(other_cols);
        }

        let outshape = outbatch;

        let a_mat_count = batch_a.iter().product::<usize>();
        let b_mat_count = batch_b.iter().product::<usize>();

        let mut out_data: Vec<f32> = vec![0.0f32; numbatches * (self_rows * other_cols)];
        for b in 0..numbatches {
            let out_offset: usize = b * (self_rows * other_cols);
            let a_off = if a_mat_count > 1 {
                b * self_rows * self_cols
            } else {
                0
            };
            let b_off = if b_mat_count > 1 {
                b * other_rows * other_cols
            } else {
                0
            };
            for i in 0..self_rows {
                let outoff = out_offset + i * other_cols;
                for k in 0..self_cols {
                    let aval = s._data[a_off + i * self_cols + k];
                    if aval == 0.0 {
                        continue; //optimzation to skip if an element is 0
                    }
                    let b_offset = b_off + k * other_cols;
                    for j in 0..other_cols {
                        out_data[outoff + j] += aval * o._data[b_offset + j];
                    }
                }
            }
        }

        drop(s);
        drop(o);
        let mut output = Tensor::new(out_data, outshape.clone());

        let s_clone = self.clone();
        let o_clone = other.clone();
        output.set_autograd(
            vec![self.clone(), other.clone()],
            Box::new(move |grad| {
                let s_req = s_clone.requires_grad();
                let o_req = o_clone.requires_grad();

                if !s_req && !o_req {
                    return;
                }

                // Prepare shapes considering 1D promotions
                let mut grad_shape = outshape.clone();
                if a_is_1D {
                    let insert_pos = if b_is_1D {
                        grad_shape.len()
                    } else {
                        grad_shape.len() - 1
                    };
                    grad_shape.insert(insert_pos, 1);
                }
                if b_is_1D {
                    grad_shape.push(1);
                }

                let grad_tensor = Tensor::new(grad.to_vec(), grad_shape);

                let s_mat = if a_is_1D {
                    let s_data = s_clone.0.borrow()._data.clone();
                    let k = s_data.len();
                    Tensor::new(s_data, vec![1, k])
                } else {
                    s_clone.clone()
                };

                let o_mat = if b_is_1D {
                    let o_data = o_clone.0.borrow()._data.clone();
                    let k = o_data.len();
                    Tensor::new(o_data, vec![k, 1])
                } else {
                    o_clone.clone()
                };

                // s_grad = grad * other^T
                if s_req {
                    let o_t = o_mat.t();
                    let s_grad_tensor = &grad_tensor * &o_t;
                    let s_data = s_grad_tensor.0.borrow()._data.clone();
                    let s_len = s_clone.numel();

                    if s_data.len() > s_len {
                        let mut reduced = vec![0.0; s_len];
                        for (i, &g) in s_data.iter().enumerate() {
                            reduced[i % s_len] += g;
                        }
                        s_clone.add_grad(&reduced);
                    } else {
                        s_clone.add_grad(&s_data);
                    }
                }

                // o_grad = self^T * grad
                if o_req {
                    let s_t = s_mat.t();
                    let o_grad_tensor = &s_t * &grad_tensor;
                    let o_data = o_grad_tensor.0.borrow()._data.clone();
                    let o_len = o_clone.numel();

                    if o_data.len() > o_len {
                        let mut reduced = vec![0.0; o_len];
                        for (i, &g) in o_data.iter().enumerate() {
                            reduced[i % o_len] += g;
                        }
                        o_clone.add_grad(&reduced);
                    } else {
                        o_clone.add_grad(&o_data);
                    }
                }
            }),
        );
        output
    }
}
//matmul
impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        &self * &other
    }
}

// scalar multiplication
impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: f32) -> Tensor {
        other * self
    }
}
impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        let o = other.0.borrow();
        let new_data: Vec<f32> = o._data.iter().map(|a| a * self).collect();
        let shape = o._shape.clone();
        drop(o);
        let mut output = Tensor::new(new_data, shape);
        let o_clone = other.clone();
        output.set_autograd(
            vec![other.clone()],
            Box::new(move |grad| {
                let o_grad: Vec<f32> = grad.iter().map(|&g| g * self).collect();
                o_clone.add_grad(&o_grad);
            }),
        );
        output
    }
}
impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, other: f32) -> Tensor {
        other * &self
    }
}
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        self * &other
    }
}
