use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyModule;
use std::collections::HashMap;
use numpy::{PyArray2};
use rand::Rng;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn neg_uniform_sample(
    train_df: &PyArray2<i64>,  // Expect a NumPy array from Python
    full_adj_list: HashMap<i64, HashMap<String, Vec<i64>>>,
    n_usr: i64
) -> PyResult<Py<PyArray2<i64>>> {

    Python::with_gil(|py| {

        unsafe {
            let interactions = train_df.as_array();  

            let mut rng = rand::thread_rng();

            // Initialize vectors for users, positive items, and negative items
            let mut users: Vec<i64> = Vec::new();
            let mut pos_items: Vec<i64> = Vec::new();
            let mut neg_items: Vec<i64> = Vec::new();

            // Iterate over the rows of the interactions (users and positive items)
            for row in interactions.outer_iter() {
                let user_id = row[0];
                let pos_item = row[1];

                // Randomly select a negative item from the user's "neg_items" list in full_adj_list
                let neg_item_list = &full_adj_list[&user_id]["neg_items"];
                let neg_item = neg_item_list[rng.gen_range(0..neg_item_list.len())];

                // Adjust positive and negative items by adding n_usr
                pos_items.push(pos_item + n_usr);
                neg_items.push(neg_item + n_usr);
                users.push(user_id);
            }

             // Create a 2D vector from users, pos_items, and neg_items
            let result_array: Vec<Vec<i64>> = (0..users.len())
                .map(|i| vec![users[i], pos_items[i], neg_items[i]])
                .collect();

            // Create a PyArray2 from the result_array using from_vec2_bound
            let result = PyArray2::from_vec2_bound(py, &result_array).unwrap(); // Correct method
            
            // Convert to the expected type for return
            Ok(result.into())
            //Ok(result)
        }
    })
}

#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(neg_uniform_sample, m)?)?;
    m.add_function(wrap_pyfunction!(neg_uniform_sample_slower, m)?)?;
    Ok(())
}