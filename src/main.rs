use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use lazy_static::lazy_static;
use ndarray::prelude::*;
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourcedIndex {
    name: String,
    source_dimensions: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BinaryArrowOperation {
    first_input_indices: Vec<String>,
    second_input_indices: Vec<String>,
    output_sourced_indices: Vec<SourcedIndex>,
    summation_sourced_indices: Vec<SourcedIndex>,
}

lazy_static! {
    static ref ARROW_EXPRESSION_REGEX: regex::Regex =
        Regex::new(r"((?:\w+ *)+), *((?:\w+ *)+)(?:->|→) *((?:\w+ *)+)").expect("valid regex");
}

fn parse_indices(indices: &str) -> Vec<String> {
    indices
        .split_whitespace()
        .map(|i| i.to_owned())
        .collect::<Vec<_>>()
}

fn parse_arrow_expression(expression: &str) -> Result<BinaryArrowOperation, String> {
    if let Some(captures) = ARROW_EXPRESSION_REGEX.captures(expression) {
        let (_, [raw_first, raw_second, raw_output]) = captures.extract();
        let first_input_indices = parse_indices(raw_first);
        let second_input_indices = parse_indices(raw_second);
        let output_indices = parse_indices(raw_output);

        let mut full_indices_set = HashSet::new();
        full_indices_set.extend(first_input_indices.clone());
        full_indices_set.extend(second_input_indices.clone());
        full_indices_set.extend(output_indices.clone());

        let mut output_indices_set = HashSet::new();
        output_indices_set.extend(output_indices.clone());

        let summation_indices_set = full_indices_set.difference(&output_indices_set);
        let mut summation_indices = summation_indices_set.into_iter().collect::<Vec<_>>();
        summation_indices.sort_by(|a, b| a.cmp(&b));

        let mut output_sourced_indices = vec![];
        for output_index in output_indices {
            let mut source_dimensions = vec![vec![], vec![]];
            for (input_no, input_indices) in [&first_input_indices, &second_input_indices]
                .iter()
                .enumerate()
            {
                for (input_dimension, input_index) in input_indices.iter().enumerate() {
                    if output_index == *input_index {
                        source_dimensions[input_no].push(input_dimension);
                    }
                }
            }
            output_sourced_indices.push(SourcedIndex {
                name: output_index,
                source_dimensions,
            });
        }

        let mut summation_sourced_indices = vec![];
        for summation_index in summation_indices {
            let mut source_dimensions = vec![vec![], vec![]];
            for (input_no, input_indices) in [&first_input_indices, &second_input_indices]
                .iter()
                .enumerate()
            {
                for (input_dimension, input_index) in input_indices.iter().enumerate() {
                    if summation_index == input_index {
                        source_dimensions[input_no].push(input_dimension);
                    }
                }
            }
            summation_sourced_indices.push(SourcedIndex {
                name: summation_index.to_string(),
                source_dimensions,
            });
        }

        let operation = BinaryArrowOperation {
            first_input_indices,
            second_input_indices,
            output_sourced_indices,
            summation_sourced_indices,
        };

        Ok(operation)
    } else {
        Err("didn't parse".to_owned())
    }
}

fn einsum(
    expression: &str,
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
) -> Result<Array<f64, IxDyn>, String> {
    let inputs = [a, b];
    let operation = parse_arrow_expression(expression)?;
    let mut index_sizes = HashMap::<String, usize>::default();
    let mut index_locations = HashMap::<String, Vec<(usize, usize)>>::default();
    for (input_no, input_indices) in [
        operation.first_input_indices.clone(),
        operation.second_input_indices.clone(),
    ]
    .iter()
    .enumerate()
    {
        // We information about what size and input dimension the index labels
        // correspond to, because when we loop over the summation and output
        // indices later, we need to know what corresponding entries of the
        // original inputs to process
        for (dimension_no, index_name) in input_indices.iter().enumerate() {
            let incoming_size = inputs[input_no].shape()[dimension_no];
            match index_sizes.entry(index_name.to_owned()) {
                Entry::Occupied(entry) => {
                    let incumbent_size = entry.get();

                    if *incumbent_size != incoming_size {
                        return Err(format!(
                            "index '{}' assigned inconsistent sizes: {} ≠ {}",
                            index_name, incumbent_size, incoming_size
                        ));
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(incoming_size);
                }
            }

            match index_locations.entry(index_name.to_owned()) {
                Entry::Occupied(mut entry) => {
                    let incumbent_locations = entry.get_mut();
                    incumbent_locations.push((input_no, dimension_no));
                }
                Entry::Vacant(entry) => {
                    entry.insert(vec![(input_no, dimension_no)]);
                }
            }
        }
    }

    let shape = operation
        .output_sourced_indices
        .iter()
        .map(|si| index_sizes.get(&*si.name).expect("known index").to_owned())
        .collect::<Vec<_>>();
    let mut output: Array<f64, IxDyn> = Array::zeros(shape);

    // use recursion to simulate nested for loops for the output indices

    fn summation_loops(
        operation: &BinaryArrowOperation,
        index_locations: &HashMap<String, Vec<(usize, usize)>>,
        first_input: &Array<f64, IxDyn>,
        second_input: &Array<f64, IxDyn>,
        summation_index_sizes: &[usize],
        current_depth: usize,
        output_index_values: &mut [usize],
        summation_index_values: &mut [usize],
        total: &mut f64,
    ) {
        if current_depth == summation_index_sizes.len() {
            // `input_index_values` is going to be the indexes that we use in
            // the "+= A[i, j] * B[j, k]" step (imagining matrix multiplication
            // as our prototypical example case). We initialize these to a
            // dummy value, then fill in the output indices (i, k in the
            // prototype example) and the summation indices (j).
            //
            // But given that we know the values of (e.g.) i, j, and k in the implied
            // nested loops, we still need to match those up with the
            // appropriate dimensions of A and B.

            let mut input_index_values = vec![
                vec![0; first_input.shape().len()],
                vec![0; second_input.shape().len()],
            ];

            // We have the output and summation indices, in the order they were
            // stored on the BinaryArrow operation.
            //
            // And we need to map them onto the corresponding input dimensions.

            for (index_value, sourced_index) in summation_index_values
                .iter()
                .zip(&operation.summation_sourced_indices)
            {
                for address in index_locations
                    .get(&sourced_index.name)
                    .expect("previously recorded")
                {
                    input_index_values[address.0][address.1] = *index_value;
                }
            }
            for (index_value, sourced_index) in output_index_values
                .iter()
                .zip(&operation.output_sourced_indices)
            {
                for address in index_locations
                    .get(&sourced_index.name)
                    .expect("previously recorded")
                {
                    input_index_values[address.0][address.1] = *index_value;
                }
            }

            *total += first_input[input_index_values[0].as_slice()]
                * second_input[input_index_values[1].as_slice()];
            return;
        }

        for i in 0..summation_index_sizes[current_depth] {
            summation_index_values[current_depth] = i;
            summation_loops(
                operation,
                index_locations,
                first_input,
                second_input,
                summation_index_sizes,
                current_depth + 1,
                output_index_values,
                summation_index_values,
                total,
            );
        }
    }

    fn output_loops(
        operation: &BinaryArrowOperation,
        index_sizes: &HashMap<String, usize>,
        index_locations: &HashMap<String, Vec<(usize, usize)>>,
        first_input: &Array<f64, IxDyn>,
        second_input: &Array<f64, IxDyn>,
        output: &mut Array<f64, IxDyn>,
        output_index_sizes: &[usize],
        current_depth: usize,
        output_index_values: &mut [usize],
    ) {
        if current_depth == output_index_sizes.len() {
            let summation_index_sizes = operation
                .summation_sourced_indices
                .iter()
                .map(|si| *index_sizes.get(&si.name).expect("size previously recorded"))
                .collect::<Vec<_>>();
            let mut summation_index_values = vec![0; operation.summation_sourced_indices.len()];
            let mut total = 0.;
            summation_loops(
                operation,
                index_locations,
                first_input,
                second_input,
                &summation_index_sizes,
                0, // launch at depth 0
                output_index_values,
                &mut summation_index_values,
                &mut total,
            );
            output[&*output_index_values] = total;
            return;
        }
        for i in 0..output_index_sizes[current_depth] {
            output_index_values[current_depth] = i;
            output_loops(
                operation,
                index_sizes,
                index_locations,
                first_input,
                second_input,
                output,
                output_index_sizes,
                current_depth + 1,
                output_index_values,
            )
        }
    }

    let output_index_sizes = operation
        .output_sourced_indices
        .iter()
        .map(|si| *index_sizes.get(&si.name).expect("size previously recorded"))
        .collect::<Vec<_>>();
    let mut output_index_values = vec![0; operation.output_sourced_indices.len()];
    output_loops(
        &operation,
        &index_sizes,
        &index_locations,
        a,
        b,
        &mut output,
        &output_index_sizes,
        0,
        &mut output_index_values,
    );

    Ok(output)
}

fn main() {
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).into_dyn();
    let b = arr2(&[[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]]).into_dyn();

    let a_ = a.clone().into_shape((3, 3)).expect("dimensions faithful");
    let b_ = b.clone().into_shape((3, 3)).expect("dimensions faithful");

    // [[ 36.,  42.,  48.],
    //  [ 81.,  96., 111.],
    //  [126., 150., 174.]
    assert_eq!(
        einsum("i j, j k → i k", &a.clone(), &b.clone()).expect("einsummed"),
        a_.dot(&b_).into_dyn()
    );

    // [[ 2.,  6., 12.],
    //  [20., 30., 42.],
    //  [56., 72., 90.]
    assert_eq!(
        einsum("i j, i j → i j", &a.clone(), &b.clone()).expect("einsummed"),
        a * b
    );
}

#[cfg(test)]
mod tests {
    use super::{
        einsum, parse_arrow_expression, parse_indices, BinaryArrowOperation, SourcedIndex,
        ARROW_EXPRESSION_REGEX,
    };
    use ndarray::{arr0, arr1, arr2, Array};

    #[test]
    fn test_arrow_expression_regex_matches() {
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j k, l m n -> i n m"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, j k -> i k"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, j k → i k"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, i j -> i j"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, i j → i j"));
    }

    #[test]
    fn test_arrow_expression_regex_groups() {
        let captures = ARROW_EXPRESSION_REGEX
            .captures("i j k, l m n -> i n m")
            .unwrap()
            .iter()
            .map(|m| m.expect("match").as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            captures,
            // TODO: could a smarter regex avoid the trailing whitespace?
            vec!["i j k, l m n -> i n m", "i j k", "l m n ", "i n m"]
        );
    }

    #[test]
    fn test_parse_indices() {
        assert_eq!(
            parse_indices("i j k"),
            vec!["i".to_owned(), "j".to_owned(), "k".to_owned()]
        );
        assert_eq!(
            parse_indices("i j k "),
            vec!["i".to_owned(), "j".to_owned(), "k".to_owned()]
        )
    }

    #[test]
    fn test_parse_arrow_expression() {
        let operation = parse_arrow_expression("i j k, l m n -> i m n").expect("parses");
        let expected = BinaryArrowOperation {
            first_input_indices: vec!["i", "j", "k"].iter().map(|&s| s.to_owned()).collect(),
            second_input_indices: vec!["l", "m", "n"].iter().map(|&s| s.to_owned()).collect(),
            output_sourced_indices: vec![
                SourcedIndex {
                    name: "i".to_owned(),
                    source_dimensions: vec![vec![0], vec![]],
                },
                SourcedIndex {
                    name: "m".to_owned(),
                    source_dimensions: vec![vec![], vec![1]],
                },
                SourcedIndex {
                    name: "n".to_owned(),
                    source_dimensions: vec![vec![], vec![2]],
                },
            ],
            summation_sourced_indices: vec![
                SourcedIndex {
                    name: "j".to_owned(),
                    source_dimensions: vec![vec![1], vec![]],
                },
                SourcedIndex {
                    name: "k".to_owned(),
                    source_dimensions: vec![vec![2], vec![]],
                },
                SourcedIndex {
                    name: "l".to_owned(),
                    source_dimensions: vec![vec![], vec![0]],
                },
            ],
        };
        assert_eq!(expected, operation);
    }

    #[test]
    fn test_einsum_inconsistent_index_size_error() {
        let a = Array::zeros((2, 3)).into_dyn();
        let b = Array::zeros((4, 5)).into_dyn();
        let result = einsum("i j, j k -> i k", &a, &b);
        assert_eq!(
            Err("index 'j' assigned inconsistent sizes: 3 ≠ 4".to_owned()),
            result
        );
    }

    #[test]
    fn test_einsum_outer_prodcut() {
        let a = arr1(&[1., 2., 3.]).into_dyn();
        let b = arr1(&[4., 5.]).into_dyn();

        let expected = arr2(&[[4., 5.], [8., 10.], [12., 15.]]).into_dyn();
        let result = einsum("a, b → a b", &a, &b).expect("einsummed");
        assert_eq!(expected, result);
    }

    // TODO? Empty right-hand side works in einops.einsum ...
    //
    // In [3]: u = torch.tensor([1, 2, 3])
    //
    // In [4]: v = torch.tensor([4, 5, 6])
    //
    // In [5]: einops.einsum(u, v, "a, a ->")
    // Out[5]: tensor(32)
    //
    // In [6]: einops.einsum(u, v, "a, a ->").shape
    // Out[6]: torch.Size([])
    #[test]
    #[ignore]
    fn test_inner_product() {
        let u = arr1(&[1., 2., 3.]).into_dyn();
        let v = arr1(&[4., 5., 6.]).into_dyn();

        let expected = arr0(32.).into_dyn();
        let result = einsum("a, a →", &u, &v).expect("einsummed");
        assert_eq!(expected, result);
    }

}

// TODO—
// • Empty right-hand side
// • More examples (cover list at the bottom of
// https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
// • Generalize to arbitrary number of inputs
// • macro_rules! so that the arbitrary inputs can be supplied as varargs
