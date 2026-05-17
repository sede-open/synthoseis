import * as React from 'react';
import { useState } from 'react';
import { UID } from './UIDComponent';
import { createSource, source } from './context';
import { useUID } from './hooks';
/**
 * UID isolation component, required for SSR and testing.
 * Wrap your application with it to guarantee UID consistency between SSR and CSR.
 * @param {String} [prefix] - prefix for all generated ids
 * @example
 * <UIDReset>
 *    <App />
 * </UIDReset/>
 * @see https://github.com/thearnica/react-uid#server-side-friendly-uid
 */
export var UIDReset = function (_a) {
    var children = _a.children, _b = _a.prefix, prefix = _b === void 0 ? '' : _b;
    var valueSource = useState(function () { return createSource(prefix); })[0];
    return React.createElement(source.Provider, { value: valueSource }, children);
};
/**
 * Creates a sub-ids for nested components, isolating from inside a branch.
 * Useful for self-contained elements or code splitting
 * @see https://github.com/thearnica/react-uid#code-splitting
 */
export var UIDFork = function (_a) {
    var children = _a.children, _b = _a.prefix, prefix = _b === void 0 ? '' : _b;
    var id = useUID();
    var valueSource = useState(function () { return createSource(id + '-' + prefix); })[0];
    return React.createElement(source.Provider, { value: valueSource }, children);
};
/**
 * UID in form of renderProps. Supports nesting and SSR. Prefer {@link useUID} hook version if possible.
 * @see https://github.com/thearnica/react-uid#server-side-friendly-uid
 * @see https://github.com/thearnica/react-uid#react-components
 * @example
 * // get UID to connect label to input
 * <UIDConsumer>
 *   {(id)} => <label htmlFor={id}><input id={id}/>}
 * </UIDConsumer>
 *
 * // get uid to generate uid for a keys in a list
 * <UIDConsumer>
 *   {(, uid)} => items.map(item => <li key={uid(item) />)}
 * </UIDConsumer>
 *
 * @see {@link useUID} - a hook version of this component
 * @see {@link UID} - not SSR compatible version
 */
export var UIDConsumer = function (_a) {
    var name = _a.name, children = _a.children;
    return (React.createElement(source.Consumer, null, function (value) { return React.createElement(UID, { name: name, idSource: value, children: children }); }));
};
