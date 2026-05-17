"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.useUIDSeed = exports.useUID = void 0;
var React = require("react");
var react_1 = require("react");
var context_1 = require("./context");
var generateUID = function (context) {
    var quartz = context || context_1.counter;
    var prefix = (0, context_1.getPrefix)(quartz);
    var id = (0, context_1.getId)(quartz);
    var uid = prefix + id;
    var gen = function (item) { return uid + quartz.uid(item); };
    return { uid: uid, gen: gen };
};
var useUIDState = function () {
    if (process.env.NODE_ENV !== 'production') {
        if (!('useContext' in React)) {
            throw new Error('Hooks API requires React 16.8+');
        }
    }
    var context = (0, react_1.useContext)(context_1.source);
    var uid = (0, react_1.useState)(function () { return generateUID(context); })[0];
    return uid;
};
/**
 * returns and unique id. SSR friendly
 * returns {String}
 * @see {@link UIDConsumer}
 * @see https://github.com/thearnica/react-uid#hooks-168
 * @example
 * const id = useUID();
 * id == 1; // for example
 */
var useUID = function () {
    var uid = useUIDState().uid;
    return uid;
};
exports.useUID = useUID;
/**
 * returns an uid generator
 * @see {@link UIDConsumer}
 * @see https://github.com/thearnica/react-uid#hooks-168
 * @example
 * const uid = useUIDSeed();
 * return (
 *  <>
 *    <label for={seed('email')}>Email: </label>
 *    <input id={seed('email')} name="email" />
 *    {data.map(item => <div key={seed(item)}>...</div>
 *  </>
 * )
 */
var useUIDSeed = function () {
    var gen = useUIDState().gen;
    return gen;
};
exports.useUIDSeed = useUIDSeed;
