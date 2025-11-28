import { DatasetEntry } from '../types/dataset';

const DB_NAME = 'golf_course_db';
const DB_VERSION = 1;
const STORE_NAME = 'dataset';

// IndexedDB instance cache
let dbInstance: IDBDatabase | null = null;

// Open/get IndexedDB connection
function openDB(): Promise<IDBDatabase> {
  if (dbInstance) {
    return Promise.resolve(dbInstance);
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);

    request.onsuccess = () => {
      dbInstance = request.result;
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
  });
}

// Save entry to IndexedDB
export async function saveToDataset(entry: DatasetEntry): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(entry);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Get all entries from IndexedDB
export async function getDatasetAsync(): Promise<DatasetEntry[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onsuccess = () => {
      const entries = request.result || [];
      // Sort by timestamp descending (newest first)
      entries.sort((a, b) => b.timestamp - a.timestamp);
      resolve(entries);
    };
    request.onerror = () => reject(request.error);
  });
}

// Synchronous version for compatibility (returns cached/empty data)
let cachedDataset: DatasetEntry[] = [];

export function getDataset(): DatasetEntry[] {
  // Return cached data synchronously
  // Async load happens in the background
  loadDatasetInBackground();
  return cachedDataset;
}

// Load dataset in background and update cache
async function loadDatasetInBackground(): Promise<void> {
  try {
    cachedDataset = await getDatasetAsync();
  } catch (e) {
    console.warn('Failed to load dataset from IndexedDB:', e);
  }
}

// Initialize cache on module load
if (typeof window !== 'undefined') {
  loadDatasetInBackground();
}

// Clear all entries
export async function clearDataset(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.clear();

    request.onsuccess = () => {
      cachedDataset = [];
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}

// Update an entry
export async function updateEntry(updatedEntry: DatasetEntry): Promise<void> {
  await saveToDataset(updatedEntry);
  // Update cache
  const index = cachedDataset.findIndex(e => e.id === updatedEntry.id);
  if (index !== -1) {
    cachedDataset[index] = updatedEntry;
  }
}

// Delete an entry
export async function deleteEntry(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.delete(id);

    request.onsuccess = () => {
      // Update cache
      cachedDataset = cachedDataset.filter(e => e.id !== id);
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}

// Calculate total size of dataset
export async function calculateDatasetSize(): Promise<number> {
  const dataset = await getDatasetAsync();
  const dataStr = JSON.stringify(dataset);
  return new Blob([dataStr]).size;
}

// Sync version for UI (uses cache)
export function calculateDatasetSizeSync(): number {
  const dataStr = JSON.stringify(cachedDataset);
  return new Blob([dataStr]).size;
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Migration: Move data from localStorage to IndexedDB (one-time)
export async function migrateFromLocalStorage(): Promise<void> {
  const LEGACY_KEY = 'golf_course_dataset';
  const legacyData = localStorage.getItem(LEGACY_KEY);

  if (legacyData) {
    try {
      const entries: DatasetEntry[] = JSON.parse(legacyData);
      for (const entry of entries) {
        await saveToDataset(entry);
      }
      // Clear localStorage after successful migration
      localStorage.removeItem(LEGACY_KEY);
      console.log(`Migrated ${entries.length} entries from localStorage to IndexedDB`);
    } catch (e) {
      console.error('Migration from localStorage failed:', e);
    }
  }
}

// Run migration on load
if (typeof window !== 'undefined') {
  migrateFromLocalStorage();
}
