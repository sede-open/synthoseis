import "@testing-library/jest-dom";

// jsdom in this vitest setup does not implement localStorage.
// Provide a minimal in-memory stub so components using getItem/setItem/clear work.
const _localStorageStore: Record<string, string> = {};
const localStorageMock: Storage = {
  getItem: (key: string) => _localStorageStore[key] ?? null,
  setItem: (key: string, value: string) => { _localStorageStore[key] = value; },
  removeItem: (key: string) => { delete _localStorageStore[key]; },
  clear: () => { Object.keys(_localStorageStore).forEach((k) => delete _localStorageStore[k]); },
  get length() { return Object.keys(_localStorageStore).length; },
  key: (index: number) => Object.keys(_localStorageStore)[index] ?? null,
};
Object.defineProperty(globalThis, "localStorage", { value: localStorageMock, writable: true });
