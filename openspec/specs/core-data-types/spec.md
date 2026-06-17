# Core Data Types Spec

## Purpose
Documents the observed behavior of the non-matrix data type families implemented in the runtime: strings/char, structs and struct arrays, cell arrays, function handles, dictionaries, datetime/duration, categorical, and table/timetable. These types are shared across the C, Python, and TypeScript runtimes (src: runtime/matlab_runtime.h, runtime/runtime_internal.h; doc: docs/feature_status.md).

## Requirements

### Requirement: String and char values
The system SHALL represent strings/char as a UTF-8 byte buffer with length and SHALL support construction from literals and conversion to numeric character codes.

#### Scenario: Build and convert a string
- **WHEN** a program creates a string literal or calls a char-to-numeric conversion
- **THEN** the system SHALL store the bytes with an explicit length (not zero-terminated) and SHALL expose character codes for numeric operations (src: runtime/matlab_runtime.h matlab_string, matlab_string_to_codes; doc: docs/feature_status.md; test: test/Run/cell_strings.m)

### Requirement: Structs and struct arrays
The system SHALL provide scalar structs with named fields (double, matrix, or nested-struct valued) and 1-D struct arrays addressed by index.

#### Scenario: Field access and struct-array indexing
- **WHEN** a program assigns/reads `s.field`, nested `s.a.b`, or `s(i).x`, or calls `isstruct`/`isfield`/`rmfield`
- **THEN** the system SHALL resolve the field by name and the element by index, growing the struct array as needed (src: runtime/runtime_internal.h matlab_struct_s; runtime/matlab_runtime.h matlab_struct_new/matlab_struct_set_f64/matlab_struct_arr_get_or_create; test: test/Run/struct_basic.m, test/Run/struct_arr_basic.m, test/Run/struct_nested.m)

### Requirement: Cell arrays
The system SHALL provide 1-D and 2-D cell arrays whose elements may independently hold scalars, matrices, or strings, with per-element kind tracking and auto-growth on out-of-bounds write.

#### Scenario: Read/write cell elements
- **WHEN** a program builds a cell with `{...}`, indexes 1-D or 2-D elements, or calls `iscell`/`numel`
- **THEN** the system SHALL store each element with a kind tag and return the correctly typed value for the addressed cell (src: runtime/matlab_runtime.h matlab_cell_new/matlab_cell_new_2d/matlab_cell_set_mat/matlab_cell_get_mat_2d/matlab_iscell; test: test/Run/cell_basic.m, test/Run/cell_2d.m, test/Run/cell_predicates.m)

### Requirement: Function handles
The system SHALL support function handles via the `@` operator, including named handles (`@sin`) and anonymous functions with captured variables.

#### Scenario: Create and call a handle
- **WHEN** a program forms `@myFunc` or `@(x) x+1` and later calls it
- **THEN** the system SHALL invoke the referenced function with any captured scalars/matrices applied (src: doc: docs/feature_status.md "Function handle operator (@)", "Anonymous function with captures"; test: test/Run/math_func_handle.m, test/Run/math_user_handle.m)

### Requirement: Dictionary
The system SHALL provide a dictionary/`containers.Map`-style associative type supporting string or numeric keys mapped to scalar or matrix values, with membership, length, and removal.

#### Scenario: Set, get, and query a dictionary
- **WHEN** a program creates a dictionary, sets entries by string or numeric key, then reads, checks `isKey`, queries `length`, or removes a key
- **THEN** the system SHALL store the key/value pair with mixed key and value types and return the stored value or membership result (src: runtime/matlab_runtime.h matlab_dict_new/matlab_dict_set_str_f64/matlab_dict_get_num_mat/matlab_dict_has_str/matlab_dict_length/matlab_dict_remove_str; test: test/Run/dict_basic.m)

### Requirement: Datetime and duration
The system SHALL provide scalar and vector datetime and duration types with constructors, unit factories, and arithmetic following civil-date semantics in UTC.

#### Scenario: Datetime arithmetic
- **WHEN** a program builds datetimes (`datetime(y,m,d)`, `datetime("now")`) and durations (`seconds`, `minutes`, `hours`, `days`, `years`) and computes `dt + dur`, `dt - dt`, or `dur ± dur`, including over vectors
- **THEN** the system SHALL return a datetime or duration result computed via Unix-epoch/civil-date math and MATLAB-default display formatting (src: runtime/matlab_runtime.h matlab_datetime_ymd/matlab_duration_seconds/matlab_datetime_add_duration/matlab_datetime_vec; test: test/Run/datetime_basic.m, test/Run/datetime_vec.m)

### Requirement: Categorical
The system SHALL provide a 1-D categorical type built from string data, storing integer category codes against a deduplicated, alphabetically sorted category-name table.

#### Scenario: Build and query categorical
- **WHEN** a program builds a categorical from strings/cell and calls `categories`, `iscategory`, `numel`, `disp`, or `==`
- **THEN** the system SHALL return the sorted category list, membership, length, display, or element-wise equality result (src: runtime/matlab_runtime.h matlab_categorical_from_strs/matlab_categorical_categories/matlab_categorical_iscategory/matlab_categorical_eq; test: test/Run/categorical_basic.m)

### Requirement: Table and timetable
The system SHALL provide a column-major table type with named variables and a timetable extension carrying a datetime row-time axis, plus CSV readers and aggregation/synchronize utilities.

#### Scenario: Build, read, and summarize a table
- **WHEN** a program builds a table (`table(...)`), reads/writes columns via `T.name`, queries `height`/`width`/`size`/`disp`, reads CSV via `readtable`, or promotes to a timetable and calls `retime`/`synchronize`/`summary`/`head`
- **THEN** the system SHALL store columns by name with per-column kind tags and return the requested column, dimension, or aggregated result (src: runtime/matlab_runtime.h matlab_table_new/matlab_table_add_column_kind/matlab_table_get_column/matlab_table_height/matlab_readtable/matlab_timetable_retime/matlab_timetable_synchronize; test: test/Run/table_basic.m, test/Run/readtable_basic.m, test/Run/timetable_retime.m)
