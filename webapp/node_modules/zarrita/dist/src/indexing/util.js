import { InvalidSelectionError } from "../errors.js";
/** Similar to python's `range` function. Supports positive ranges only. */
export function* range(start, stop, step = 1) {
    if (stop === undefined) {
        stop = start;
        start = 0;
    }
    for (let i = start; i < stop; i += step) {
        yield i;
    }
}
/**
 * python-like itertools.product generator
 * https://gist.github.com/cybercase/db7dde901d7070c98c48
 */
export function* product(...iterables) {
    if (iterables.length === 0) {
        return;
    }
    // make a list of iterators from the iterables
    const iterators = iterables.map((it) => it[Symbol.iterator]());
    const results = iterators.map((it) => it.next());
    if (results.some((r) => r.done)) {
        throw new Error("Input contains an empty iterator.");
    }
    for (let i = 0;;) {
        if (results[i].done) {
            // reset the current iterator
            iterators[i] = iterables[i][Symbol.iterator]();
            results[i] = iterators[i].next();
            // advance, and exit if we've reached the end
            if (++i >= iterators.length) {
                return;
            }
        }
        else {
            // @ts-expect-error - TS can't infer this
            yield results.map(({ value }) => value);
            i = 0;
        }
        results[i] = iterators[i].next();
    }
}
// https://github.com/python/cpython/blob/263c0dd16017613c5ea2fbfc270be4de2b41b5ad/Objects/sliceobject.c#L376-L519
export function sliceIndices({ start, stop, step }, length) {
    if (step === 0) {
        throw new InvalidSelectionError("slice step cannot be zero");
    }
    step = step ?? 1;
    const stepIsNegative = step < 0;
    /* Find lower and upper bounds for start and stop. */
    const [lower, upper] = stepIsNegative ? [-1, length - 1] : [0, length];
    /* Compute start. */
    if (start === null) {
        start = stepIsNegative ? upper : lower;
    }
    else {
        if (start < 0) {
            start += length;
            if (start < lower) {
                start = lower;
            }
        }
        else if (start > upper) {
            start = upper;
        }
    }
    /* Compute stop. */
    if (stop === null) {
        stop = stepIsNegative ? lower : upper;
    }
    else {
        if (stop < 0) {
            stop += length;
            if (stop < lower) {
                stop = lower;
            }
        }
        else if (stop > upper) {
            stop = upper;
        }
    }
    return [start, stop, step];
}
function toInt(value) {
    if (value == null)
        return null;
    if (typeof value === "bigint") {
        if (value > Number.MAX_SAFE_INTEGER || value < Number.MIN_SAFE_INTEGER) {
            throw new InvalidSelectionError(`Cannot safely convert ${value} to a number. Value exceeds Number.MAX_SAFE_INTEGER.`);
        }
        return Number(value);
    }
    return value;
}
export function slice(start, stop, step = null) {
    if (stop === undefined) {
        stop = start;
        start = null;
    }
    return {
        start: toInt(start),
        stop: toInt(stop),
        step: toInt(step),
    };
}
/** @category Utility */
export function select(arr, selection) {
    let names = arr.dimensionNames;
    if (!names) {
        throw new InvalidSelectionError("Array does not have dimension_names in its metadata.");
    }
    for (let key of Object.keys(selection)) {
        if (!names.includes(key)) {
            throw new InvalidSelectionError(`Unknown dimension name: "${key}". Available dimensions: ${names.map((n) => `"${n}"`).join(", ")}`);
        }
    }
    return names.map((name) => selection[name] ?? null);
}
/** Built-in "queue" for awaiting promises. */
export function createQueue() {
    const promises = [];
    return {
        add: (fn) => promises.push(fn()),
        onIdle: () => Promise.all(promises),
    };
}
//# sourceMappingURL=util.js.map