'use client';

import { useState, useEffect } from 'react';
import { Camera, Check, X, Package, Trash2, Download } from 'lucide-react';
import { Button } from './ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';
import { toast } from 'sonner';

interface ClassificationEntry {
  id: string;
  imageData: string;
  label: 'golf' | 'not_golf';
  timestamp: string;
  location: {
    lat: number;
    lng: number;
  };
}

interface ClassificationCaptureProps {
  mapCenter: { lat: number; lng: number };
  capturedImageData?: string;
  onTriggerCapture: () => void;
}

// IndexedDB helpers for classification dataset
const DB_NAME = 'classification_dataset_db';
const STORE_NAME = 'classification_entries';

async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
  });
}

async function saveClassificationEntry(entry: ClassificationEntry): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(entry);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

async function getClassificationDataset(): Promise<ClassificationEntry[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

async function clearClassificationDataset(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.clear();
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

export function ClassificationCapture({
  mapCenter,
  capturedImageData,
  onTriggerCapture,
}: ClassificationCaptureProps) {
  const [currentImage, setCurrentImage] = useState<string | undefined>();
  const [golfCount, setGolfCount] = useState(0);
  const [notGolfCount, setNotGolfCount] = useState(0);
  const [isLabeling, setIsLabeling] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Load dataset counts
  useEffect(() => {
    const loadCounts = async () => {
      const dataset = await getClassificationDataset();
      setGolfCount(dataset.filter(e => e.label === 'golf').length);
      setNotGolfCount(dataset.filter(e => e.label === 'not_golf').length);
    };
    loadCounts();
  }, []);

  // Sync captured image from props
  useEffect(() => {
    if (capturedImageData) {
      setCurrentImage(capturedImageData);
      toast.success('📸 Image captured! Label it below');
    }
  }, [capturedImageData]);

  const handleLabel = async (label: 'golf' | 'not_golf') => {
    if (!currentImage) return;

    setIsLabeling(true);
    try {
      const entry: ClassificationEntry = {
        id: `cls_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        imageData: currentImage,
        label,
        timestamp: new Date().toISOString(),
        location: {
          lat: mapCenter.lat,
          lng: mapCenter.lng,
        },
      };

      await saveClassificationEntry(entry);

      // Update counts
      if (label === 'golf') {
        setGolfCount(prev => prev + 1);
        toast.success('✅ Labeled as Golf Course');
      } else {
        setNotGolfCount(prev => prev + 1);
        toast.success('❌ Labeled as Not Golf');
      }

      // Clear current image for next capture
      setCurrentImage(undefined);
    } catch (error) {
      toast.error('Failed to save label');
      console.error(error);
    } finally {
      setIsLabeling(false);
    }
  };

  const handleClearClick = () => {
    setShowClearConfirm(true);
  };

  const handleConfirmClear = async () => {
    await clearClassificationDataset();
    setGolfCount(0);
    setNotGolfCount(0);
    setShowClearConfirm(false);
    toast.success('Classification dataset cleared');
  };

  const handleExport = async () => {
    try {
      const dataset = await getClassificationDataset();
      if (dataset.length === 0) {
        toast.error('No data to export');
        return;
      }

      // Create ZIP with golf/ and not_golf/ folders
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();

      const golfFolder = zip.folder('golf');
      const notGolfFolder = zip.folder('not_golf');

      for (const entry of dataset) {
        const folder = entry.label === 'golf' ? golfFolder : notGolfFolder;
        if (folder) {
          // Convert base64 to blob
          const base64Data = entry.imageData.split(',')[1];
          const filename = `${entry.id}.jpg`;
          folder.file(filename, base64Data, { base64: true });
        }
      }

      // Create labels.csv
      const csvContent = 'filename,label,lat,lng,timestamp\n' +
        dataset.map(e =>
          `${e.label}/${e.id}.jpg,${e.label},${e.location.lat},${e.location.lng},${e.timestamp}`
        ).join('\n');
      zip.file('labels.csv', csvContent);

      // Download
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `classification_dataset_${new Date().toISOString().split('T')[0]}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(`Exported ${dataset.length} images`);
    } catch (error) {
      toast.error('Export failed');
      console.error(error);
    }
  };

  const totalCount = golfCount + notGolfCount;

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-slate-200">🏷️ Classification Dataset</h3>
        <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded">
          BINARY
        </span>
      </div>

      <p className="text-slate-400 text-sm mb-4">
        Capture images and label them as Golf or Not Golf for classifier training.
      </p>

      {/* Capture Button */}
      {!currentImage && (
        <Button
          onClick={onTriggerCapture}
          className="w-full bg-blue-500 hover:bg-blue-600 mb-4"
        >
          <Camera className="size-4 mr-2" />
          Capture Image
        </Button>
      )}

      {/* Labeling Panel */}
      {currentImage && (
        <div className="bg-slate-700/30 rounded-lg p-4 mb-4 border border-slate-600/30">
          {/* Image Preview */}
          <div className="aspect-video rounded-lg overflow-hidden mb-4 bg-slate-800">
            <img
              src={currentImage}
              alt="Captured"
              className="w-full h-full object-cover"
            />
          </div>

          {/* Label Buttons */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={() => handleLabel('golf')}
              disabled={isLabeling}
              className="h-9 bg-emerald-600 hover:bg-emerald-700 text-sm font-medium"
            >
              <Check className="size-4 mr-1" />
              Golf
            </Button>
            <Button
              onClick={() => handleLabel('not_golf')}
              disabled={isLabeling}
              className="h-9 bg-red-600 hover:bg-red-700 text-sm font-medium"
            >
              <X className="size-4 mr-1" />
              Not Golf
            </Button>
          </div>

          {/* Skip button */}
          <Button
            onClick={() => setCurrentImage(undefined)}
            variant="ghost"
            className="w-full mt-2 text-slate-400 hover:text-slate-200"
          >
            Skip this image
          </Button>
        </div>
      )}

      {/* Dataset Stats */}
      <div className="bg-slate-700/30 rounded-lg p-3 space-y-2 mb-4">
        <div className="text-sm text-slate-400 mb-2">📊 Dataset Statistics</div>

        <div className="flex justify-between text-sm">
          <span className="text-emerald-400">Golf Courses:</span>
          <span className="text-slate-200 font-medium">{golfCount}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-red-400">Not Golf:</span>
          <span className="text-slate-200 font-medium">{notGolfCount}</span>
        </div>

        <div className="border-t border-slate-600/30 pt-2 mt-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-300">Total:</span>
            <span className="text-slate-200 font-bold">{totalCount}</span>
          </div>
        </div>

        {/* Balance indicator */}
        {totalCount > 0 && (
          <div className="mt-2">
            <div className="text-xs text-slate-400 mb-1">Class Balance</div>
            <div className="h-2 bg-slate-600 rounded-full overflow-hidden flex">
              <div
                className="bg-emerald-500 h-full"
                style={{ width: `${(golfCount / totalCount) * 100}%` }}
              />
              <div
                className="bg-red-500 h-full"
                style={{ width: `${(notGolfCount / totalCount) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>{((golfCount / totalCount) * 100).toFixed(0)}% golf</span>
              <span>{((notGolfCount / totalCount) * 100).toFixed(0)}% not golf</span>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="space-y-2">
        <Button
          onClick={handleExport}
          disabled={totalCount === 0}
          className="w-full bg-slate-600 text-white hover:bg-slate-500 disabled:bg-slate-700 disabled:text-slate-500"
        >
          <Download className="size-4 mr-2" />
          Export Dataset
        </Button>

        <Button
          onClick={handleClearClick}
          disabled={totalCount === 0}
          className="w-full bg-slate-600 text-white hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500"
        >
          <Trash2 className="size-4 mr-2" />
          Clear Dataset
        </Button>
      </div>

      {/* Clear Dataset Confirmation Dialog */}
      <Dialog open={showClearConfirm} onOpenChange={setShowClearConfirm}>
        <DialogContent className="bg-slate-800 border-slate-700 text-slate-200">
          <DialogHeader>
            <DialogTitle>Clear Dataset?</DialogTitle>
            <DialogDescription className="text-slate-400">
              Are you sure you want to clear all {totalCount} labeled images? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowClearConfirm(false)}
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirmClear}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <Trash2 className="size-4 mr-2" />
              Clear All
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
