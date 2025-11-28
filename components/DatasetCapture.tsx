'use client';

import { useState, useEffect } from 'react';
import { Package, Trash2, Paintbrush, Camera, Save, Calendar, MapPin, X } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';
import { SegmentationResult } from '../app/page';
import { DatasetEntry } from '../types/dataset';
import { saveToDataset, updateEntry, getDatasetAsync, clearDataset, calculateDatasetSizeSync, formatBytes } from '../utils/datasetStorage';
import { toast } from 'sonner';
import { ExportDatasetDialog } from './ExportDatasetDialog';
import { AnnotationCanvas } from './AnnotationCanvas';
import { DatasetViewer } from './DatasetViewer';

interface DatasetCaptureProps {
  mapCenter: { lat: number; lng: number };
  segmentationResult: SegmentationResult | null;
  capturedImageData?: string;
  workflowMode: 'dataset' | 'segmentation';
  onTriggerCapture: () => void;
}

export function DatasetCapture({
  mapCenter,
  segmentationResult,
  capturedImageData,
  workflowMode,
  onTriggerCapture,
}: DatasetCaptureProps) {
  const [savedCount, setSavedCount] = useState(0);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [showAnnotationTool, setShowAnnotationTool] = useState(false);
  const [editingEntry, setEditingEntry] = useState<DatasetEntry | null>(null);
  const [currentCapturedImage, setCurrentCapturedImage] = useState<string | undefined>(capturedImageData);
  const [courseName, setCourseName] = useState('');
  const [groundTruthAnnotation, setGroundTruthAnnotation] = useState<string | undefined>();
  const [isSaving, setIsSaving] = useState(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);
  const [datasetRefreshTrigger, setDatasetRefreshTrigger] = useState(0);
  const [datasetSize, setDatasetSize] = useState('0 Bytes');

  // Load initial dataset count
  useEffect(() => {
    const loadDataset = async () => {
      const dataset = await getDatasetAsync();
      setSavedCount(dataset.length);
      setDatasetSize(formatBytes(calculateDatasetSizeSync()));
    };
    loadDataset();
  }, [datasetRefreshTrigger]);

  // Sync capturedImageData from props to local state
  useEffect(() => {
    if (capturedImageData) {
      setCurrentCapturedImage(capturedImageData);
      toast.success('📸 Image captured!');
    }
  }, [capturedImageData]);

  const handleCapture = () => {
    // Trigger the map capture
    onTriggerCapture();
  };

  const handleDiscardClick = () => {
    setShowDiscardConfirm(true);
  };

  const handleConfirmDiscard = () => {
    setCurrentCapturedImage(undefined);
    setCourseName('');
    setGroundTruthAnnotation(undefined);
    setShowDiscardConfirm(false);
    toast.success('Current entry discarded');
  };

  const handleOpenAnnotationTool = () => {
    if (!currentCapturedImage) {
      toast.error('Capture an image first');
      return;
    }
    setShowAnnotationTool(true);
  };

  const handleSave = async () => {
    if (!courseName.trim()) {
      toast.error('Please enter a course name');
      return;
    }

    setIsSaving(true);

    try {
      // Create dataset entry
      const entry: DatasetEntry = {
        id: `course_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        filename: `${courseName}_${Date.now()}`,
        courseName: courseName,
        timestamp: new Date().toISOString(),
        location: {
          name: courseName,
          lat: mapCenter.lat,
          lng: mapCenter.lng,
          zoomLevel: 17,
        },
        captureBox: {
          width: 1664,
          height: 1024,
          bounds: {
            north: mapCenter.lat + 0.003,
            south: mapCenter.lat - 0.003,
            east: mapCenter.lng + 0.005,
            west: mapCenter.lng - 0.005,
          },
        },
        images: {
          satellite: currentCapturedImage,
        },
      };

      // Add segmentation result if available
      if (segmentationResult) {
        entry.images.mask = segmentationResult.overlayData;
        entry.segmentation = {
          model: 'ResNet50_UNet',
          version: '1.0',
          hasPrediction: true,
          hasGroundTruth: !!groundTruthAnnotation,
          classDistribution: segmentationResult.statistics,
        };
      }

      // Add ground truth if available
      if (groundTruthAnnotation) {
        entry.images.groundTruth = groundTruthAnnotation;
        if (!entry.segmentation) {
          entry.segmentation = {
            model: 'Manual',
            version: '1.0',
            hasPrediction: false,
            hasGroundTruth: true,
            classDistribution: {},
          };
        }
      }

      // Save to dataset (async - IndexedDB)
      await saveToDataset(entry);
      const dataset = await getDatasetAsync();
      setSavedCount(dataset.length);
      setDatasetRefreshTrigger(prev => prev + 1);
      toast.success('✅ Saved to dataset!');

      // Reset
      setCourseName('');
      setCurrentCapturedImage(undefined);
      setGroundTruthAnnotation(undefined);

    } catch (error) {
      toast.error('❌ Failed to save');
      console.error(error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleEditEntryAnnotation = (entry: DatasetEntry) => {
    setEditingEntry(entry);
    setShowAnnotationTool(true);
  };

  const handleSaveAnnotation = async (annotationData: string) => {
    if (editingEntry) {
      // Editing a saved entry
      const updatedEntry: DatasetEntry = {
        ...editingEntry,
        images: {
          ...editingEntry.images,
          groundTruth: annotationData,
        },
        segmentation: {
          ...(editingEntry.segmentation || {
            model: 'Manual',
            version: '1.0',
            hasPrediction: false,
            classDistribution: {},
          }),
          hasGroundTruth: true,
        },
      };

      await updateEntry(updatedEntry);
      const dataset = await getDatasetAsync();
      setSavedCount(dataset.length);
      setDatasetRefreshTrigger(prev => prev + 1);
      setEditingEntry(null);
      setShowAnnotationTool(false);
      toast.success('✅ Annotation updated!');
    } else {
      // Annotating current capture - SAVE IMMEDIATELY but KEEP current entry visible
      setShowAnnotationTool(false);
      setIsSaving(true);
      toast.info('💾 Saving to dataset...');

      try {
        // Generate a course name if not provided
        const finalCourseName = courseName.trim() || `Golf Course ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`;

        // Create dataset entry with annotation
        const entry: DatasetEntry = {
          id: `course_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          filename: `${finalCourseName}_${Date.now()}`,
          courseName: finalCourseName,
          timestamp: new Date().toISOString(),
          location: {
            name: finalCourseName,
            lat: mapCenter.lat,
            lng: mapCenter.lng,
            zoomLevel: 17,
          },
          captureBox: {
            width: 1664,
            height: 1024,
            bounds: {
              north: mapCenter.lat + 0.003,
              south: mapCenter.lat - 0.003,
              east: mapCenter.lng + 0.005,
              west: mapCenter.lng - 0.005,
            },
          },
          images: {
            satellite: currentCapturedImage,
            groundTruth: annotationData,
          },
          segmentation: {
            model: 'Manual',
            version: '1.0',
            hasPrediction: !!segmentationResult,
            hasGroundTruth: true,
            classDistribution: segmentationResult?.statistics || {},
          },
        };

        // Add segmentation mask if available
        if (segmentationResult) {
          entry.images.mask = segmentationResult.overlayData;
        }

        // Save to dataset (async - IndexedDB)
        await saveToDataset(entry);

        // Get updated count
        const dataset = await getDatasetAsync();
        setSavedCount(dataset.length);

        // Trigger dataset viewer refresh
        setDatasetRefreshTrigger(prev => prev + 1);

        // Keep current entry - just update the annotation state
        // User can manually discard when ready for next capture
        setGroundTruthAnnotation(annotationData);
        if (!courseName.trim()) {
          setCourseName(finalCourseName);
        }

        toast.success('✅ Saved to dataset! Entry still visible for reference.');
      } catch (error) {
        console.error('Failed to save:', error);
        toast.error('❌ Failed to save: ' + (error instanceof Error ? error.message : 'Unknown error'));
        // Store annotation so user can try again
        setGroundTruthAnnotation(annotationData);
      } finally {
        setIsSaving(false);
      }
    }
  };


  const handleClearDataset = async () => {
    if (window.confirm('Are you sure you want to clear all saved data? This cannot be undone.')) {
      await clearDataset();
      setSavedCount(0);
      setDatasetRefreshTrigger(prev => prev + 1);
      toast.success('Dataset cleared');
    }
  };

  // datasetSize is now managed via state and updated in useEffect

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-slate-200">💾 Dataset Capture</h3>
        <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded">
          {workflowMode === 'dataset' ? 'ACTIVE' : 'AVAILABLE'}
        </span>
      </div>
      
      <div className="space-y-4">
        <p className="text-slate-400 text-sm">
          {workflowMode === 'dataset' 
            ? 'Capture golf course imagery to build your dataset'
            : 'Save current view to build your dataset'
          }
        </p>

        {/* Capture Button - Only in dataset mode */}
        {workflowMode === 'dataset' && (
          <Button
            onClick={handleCapture}
            disabled={!!currentCapturedImage}
            className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-slate-600 disabled:text-slate-400"
          >
            <Camera className="size-4 mr-2" />
            Capture Current View
          </Button>
        )}

        {/* Current Entry Panel */}
        {currentCapturedImage && (
          <div className="space-y-3 border-t border-slate-600/30 pt-3 mt-3">
            <div className="flex items-center justify-between">
              <h4 className="text-slate-300 text-sm font-medium">Current Entry</h4>
              <Button
                onClick={handleDiscardClick}
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-white hover:bg-red-900/50 hover:text-red-400"
              >
                <X className="size-4" />
              </Button>
            </div>

            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/30">
              {/* Thumbnail and Info */}
              <div className="flex items-start gap-3 mb-3">
                <div className="size-16 rounded overflow-hidden flex-shrink-0 bg-slate-800">
                  <img
                    src={currentCapturedImage}
                    alt="Captured"
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
                    <Calendar className="size-3" />
                    <span>{new Date().toLocaleDateString()}</span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <MapPin className="size-3" />
                    <span>
                      {mapCenter.lat.toFixed(4)}, {mapCenter.lng.toFixed(4)}
                    </span>
                  </div>

                  {/* Status badges */}
                  <div className="flex gap-1 mt-2">
                    {groundTruthAnnotation && (
                      <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded">
                        Annotated
                      </span>
                    )}
                    {segmentationResult && (
                      <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded">
                        Predicted
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Course Name Input */}
              <div className="mb-3">
                <Label htmlFor="courseName" className="text-slate-300 text-xs mb-1.5 block">
                  Course Name *
                </Label>
                <Input
                  id="courseName"
                  value={courseName}
                  onChange={(e) => setCourseName(e.target.value)}
                  placeholder="Pebble Beach - Hole 7"
                  className="bg-slate-700 border-slate-600 text-slate-200 h-8 text-sm"
                />
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <Button
                  onClick={handleOpenAnnotationTool}
                  variant="outline"
                  className="flex-1 h-8 border-slate-600 text-black hover:bg-purple-900 hover:text-purple-400"
                >
                  <Paintbrush className="size-3 mr-1.5" />
                  {groundTruthAnnotation ? 'Edit' : 'Annotate'}
                </Button>

                <Button
                  onClick={handleSave}
                  disabled={!courseName.trim() || isSaving}
                  className="flex-1 h-8 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-600 disabled:text-slate-400"
                >
                  {isSaving ? (
                    <>
                      <div className="size-3 border-2 border-white border-t-transparent rounded-full animate-spin mr-1.5" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="size-3 mr-1.5" />
                      Save
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Location Display */}
        <div className="bg-slate-700/30 rounded p-2 text-xs text-slate-400">
          📍 Lat: {mapCenter.lat.toFixed(4)}, Lng: {mapCenter.lng.toFixed(4)}
        </div>

        {/* Dataset Info */}
        <div className="bg-slate-700/30 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">📊 Current Session:</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-300">Images saved:</span>
            <span className="text-slate-200 font-medium">{savedCount}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-300">Size:</span>
            <span className="text-slate-200 font-medium">{datasetSize}</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2">
          <Button
            onClick={() => setShowExportDialog(true)}
            disabled={savedCount === 0}
            variant="outline"
            className="w-full border-slate-600 text-black hover:bg-slate-700"
          >
            <Package className="size-4 mr-2" />
            Export Dataset
          </Button>

          <Button
            onClick={handleClearDataset}
            disabled={savedCount === 0}
            variant="outline"
            className="w-full border-slate-600 text-black hover:bg-slate-700 hover:text-red-400"
          >
            <Trash2 className="size-4 mr-2" />
            Clear Session
          </Button>
        </div>

        {/* Dataset Viewer - View and Edit Saved Annotations */}
        <div className="border-t border-slate-600/30 pt-4 mt-4">
          <DatasetViewer
            onEditAnnotation={handleEditEntryAnnotation}
            onDatasetChange={() => setDatasetRefreshTrigger(prev => prev + 1)}
            refreshTrigger={datasetRefreshTrigger}
          />
        </div>
      </div>

      {/* Export Dialog */}
      <ExportDatasetDialog
        open={showExportDialog}
        onOpenChange={setShowExportDialog}
      />

      {/* Annotation Tool */}
      {showAnnotationTool && (
        <AnnotationCanvas
          imageData={editingEntry ? editingEntry.images.satellite! : currentCapturedImage!}
          initialAnnotation={editingEntry ? editingEntry.images.groundTruth : groundTruthAnnotation}
          onSave={handleSaveAnnotation}
          onClose={() => {
            setShowAnnotationTool(false);
            setEditingEntry(null);
          }}
        />
      )}

      {/* Discard Confirmation Dialog */}
      <Dialog open={showDiscardConfirm} onOpenChange={setShowDiscardConfirm}>
        <DialogContent className="bg-slate-800 border-slate-700 text-slate-200">
          <DialogHeader>
            <DialogTitle>Discard Current Entry?</DialogTitle>
            <DialogDescription className="text-slate-400">
              Are you sure you want to discard this captured image? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowDiscardConfirm(false)}
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirmDiscard}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <X className="size-4 mr-2" />
              Discard
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}