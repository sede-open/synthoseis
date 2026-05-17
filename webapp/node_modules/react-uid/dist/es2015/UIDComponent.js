var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import * as React from 'react';
import { counter, getId, getPrefix } from './context';
// --------------------------------------------
var prefixId = function (id, prefix, name) {
    var uid = prefix + id;
    return String(name ? name(uid) : uid);
};
/**
 * @deprecated
 * UID in form of renderProps (not SSR friendly)
 * @see https://github.com/thearnica/react-uid#react-components
 * @example
 * // get UID to connect label to input
 * <UID>
 *   {(id)} => <label htmlFor={id}><input id={id}/>}
 * </UID>
 *
 * // get uid to generate uid for a keys in a list
 * <UID>
 *   {(, uid)} => items.map(item => <li key={uid(item) />)}
 * </UID>
 */
var UID = /** @class */ (function (_super) {
    __extends(UID, _super);
    function UID() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.state = {
            quartz: _this.props.idSource || counter,
            prefix: getPrefix(_this.props.idSource),
            id: getId(_this.props.idSource || counter),
        };
        _this.uid = function (item) {
            return prefixId(_this.state.id + '-' + _this.state.quartz.uid(item), _this.state.prefix, _this.props.name);
        };
        return _this;
    }
    UID.prototype.render = function () {
        var _a = this.props, children = _a.children, name = _a.name;
        var _b = this.state, id = _b.id, prefix = _b.prefix;
        return children(prefixId(id, prefix, name), this.uid);
    };
    return UID;
}(React.Component));
export { UID };
