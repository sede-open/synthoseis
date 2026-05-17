# elementary-circuits-directed-graph

An implementation of the Johnson's circuit finding algorithm [1].

[1] Donald B. Johnson, Finding all the elementary circuits of a directed
    graph, SIAM Journal on Computing, 1975.

## Example

```javascript
var findCircuits = require("elementary-circuits-directed-graph");

//       V4      V2
// +-<---o---<---o---<--+
// |             |      |
// o V0          ^      o V3
// |           V1|      |
// +------>------o--->--+

var adjacencyList = [
  [1],
  [2, 3],
  [4],
  [2],
  [0]
]

console.log(findCircuits(adjacencyList))

// returns [[0, 1, 2, 4, 0], [0, 1, 3, 2, 4, 0]]
```

Optionally, one can define a callback to manage the result.
```javascript
// reusing the same adjacencyList as before
var counter = 0;
function increment() {
    counter += 1;
}
findCircuits(adjacencyList, increment);
console.log(counter)

// return 2
```
This is especially useful if there are too many elementary circuits
to store in memory. Using a callback, they can be saved to disk instead.

## Install

npm install elementary-circuits-directed-graph

## API

### `require("elementary-circuits-directed-graph")(adjacencyList, callback)`
Finds all the elementary circuits of a directed graph using

* `adjacencyList` is an array of lists representing the directed edges of the graph
* `callback` is an optional function that will be called each time an elementary circuit is found.

**Returns** An array of arrays representing the elementary circuits if no callback was defined.

## Credits
(c) 2018 Antoine Roy-Gobeil. MIT License.
