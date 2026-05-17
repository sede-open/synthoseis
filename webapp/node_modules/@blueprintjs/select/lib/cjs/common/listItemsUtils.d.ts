/**
 * The reserved type of the "Create Item" option in item lists. This is intended
 * not to conflict with any custom item type `T` that might be used in  item
 * list.
 */
export interface CreateNewItem {
    __blueprintCreateNewItemBrand: "blueprint-create-new-item";
}
/** Returns an instance of a "Create Item" object. */
export declare function getCreateNewItem(): CreateNewItem;
/**
 * Type guard returning `true` if the provided item (e.g. the current
 * `activeItem`) is a "Create Item" option.
 */
export declare function isCreateNewItem<T>(item: T | CreateNewItem | null | undefined): item is CreateNewItem;
/**
 * Returns the type of the the current active item. This will be a no-op unless
 * the `activeItem` is `undefined` or a "Create Item" option, in which case
 * `null` will be returned instead.
 */
export declare function getActiveItem<T>(activeItem: T | CreateNewItem | null | undefined): T | null;
