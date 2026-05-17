"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UIDConsumer = exports.UIDFork = exports.UIDReset = void 0;
var React = require("react");
var react_1 = require("react");
var UIDComponent_1 = require("./UIDComponent");
var context_1 = require("./context");
var hooks_1 = require("./hooks");
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
var UIDReset = function (_a) {
    var children = _a.children, _b = _a.prefix, prefix = _b === void 0 ? '' : _b;
    var valueSource = (0, react_1.useState)(function () { return (0, context_1.createSource)(prefix); })[0];
    return React.createElement(context_1.source.Provider, { value: valueSource }, children);
};
exports.UIDReset = UIDReset;
/**
 * Creates a sub-ids for nested components, isolating from inside a branch.
 * Useful for self-contained elements or code splitting
 * @see https://github.com/thearnica/react-uid#code-splitting
 */
var UIDFork = function (_a) {
    var children = _a.children, _b = _a.prefix, prefix = _b === void 0 ? '' : _b;
    var id = (0, hooks_1.useUID)();
    var valueSource = (0, react_1.useState)(function () { return (0, context_1.createSource)(id + '-' + prefix); })[0];
    return React.createElement(context_1.source.Provider, { value: valueSource }, children);
};
exports.UIDFork = UIDFork;
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
var UIDConsumer = function (_a) {
    var name = _a.name, children = _a.children;
    return (React.createElement(context_1.source.Consumer, null, function (value) { return React.createElement(UIDComponent_1.UID, { name: name, idSource: value, children: children }); }));
};
exports.UIDConsumer = UIDConsumer;
