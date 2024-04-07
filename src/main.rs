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
    for (input_no, input_indices) in [
        operation.first_input_indices,
        operation.second_input_indices,
    ]
    .iter()
    .enumerate()
    {
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
        first_input: &Array<f64, IxDyn>,
        second_input: &Array<f64, IxDyn>,
        summation_index_sizes: &[usize],
        current_depth: usize,
        output_indices: &mut [usize],
        summation_indices: &mut [usize],
        total: &mut f64,
    ) {
        if current_depth == summation_index_sizes.len() {
            // TODO: multiply and sum here
            // *total += first_input[...] * second_input[...]
        }

        for i in 0..summation_index_sizes[current_depth] {
            summation_indices[current_depth] = i;
            summation_loops(
                first_input,
                second_input,
                summation_index_sizes,
                current_depth,
                output_indices,
                summation_indices,
                total,
            );
        }
    }

    fn output_index_loops(
        first_input: &Array<f64, IxDyn>,
        second_input: &Array<f64, IxDyn>,
        output: &mut Array<f64, IxDyn>,
        output_index_sizes: &[usize],
        current_depth: usize,
        output_indices: &mut [usize],
    ) {
        if current_depth == output_index_sizes.len() {
            let mut total = 0.;

            output[&*output_indices] = total;
        }
        for i in 0..output_index_sizes[current_depth] {
            output_indices[current_depth] = i;
            output_index_loops(
                first_input,
                second_input,
                output,
                output_index_sizes,
                current_depth + 1,
                output_indices,
            )
        }
    }

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
    use ndarray::Array;

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
}
