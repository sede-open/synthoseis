"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getPrefix = exports.getId = exports.source = exports.counter = exports.createSource = void 0;
var React = require("react");
var uid_1 = require("./uid");
var createSource = function (prefix) {
    if (prefix === void 0) { prefix = ''; }
    return ({
        value: 1,
        prefix: prefix,
        uid: (0, uid_1.generateUID)(),
    });
};
exports.createSource = createSource;
exports.counter = (0, exports.createSource)();
exports.source = React.createContext((0, exports.createSource)());
var getId = function (source) { return source.value++; };
exports.getId = getId;
var getPrefix = function (source) { return (source ? source.prefix : ''); };
exports.getPrefix = getPrefix;
