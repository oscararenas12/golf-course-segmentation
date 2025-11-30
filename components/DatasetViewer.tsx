'use client';

import { useState, useEffect } from 'react';
import { DatasetEntry } from '../types/dataset';
import { getDatasetAsync, deleteEntry, updateEntry } from '../utils/datasetStorage';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';
import { ScrollArea } from './ui/scroll-area';
import { Eye, Pencil, MapPin, Calendar, Image as ImageIcon, Trash2, Save } from 'lucide-react';
import { toast } from 'sonner';

interface DatasetViewerProps {
  onEditAnnotation: (entry: DatasetEntry) => void;
  onDatasetChange?: () => void;
  refreshTrigger?: number; // Increment this to trigger a refresh
}

export function DatasetViewer({ onEditAnnotation, onDatasetChange, refreshTrigger }: DatasetViewerProps) {
  const [dataset, setDataset] = useState<DatasetEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Refresh dataset when refreshTrigger changes
  useEffect(() => {
    const loadDataset = async () => {
      setIsLoading(true);
      try {
        const data = await getDatasetAsync();
        setDataset(data);
      } catch (e) {
        console.error('Failed to load dataset:', e);
      } finally {
        setIsLoading(false);
      }
    };
    loadDataset();
  }, [refreshTrigger]);

  const [selectedEntry, setSelectedEntry] = useState<DatasetEntry | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [entryToDelete, setEntryToDelete] = useState<DatasetEntry | null>(null);
  const [editedCourseName, setEditedCourseName] = useState('');

  const handleViewEntry = (entry: DatasetEntry) => {
    setSelectedEntry(entry);
    setEditedCourseName(entry.courseName);
    setShowPreview(true);
  };

  const handleEditAnnotation = () => {
    if (selectedEntry) {
      setShowPreview(false);
      onEditAnnotation(selectedEntry);
    }
  };

  const handleDeleteClick = (entry: DatasetEntry) => {
    setEntryToDelete(entry);
    setShowDeleteConfirm(true);
  };

  const handleConfirmDelete = async () => {
    if (entryToDelete) {
      await deleteEntry(entryToDelete.id);
      const data = await getDatasetAsync();
      setDataset(data);
      setShowDeleteConfirm(false);
      setEntryToDelete(null);
      toast.success('Entry deleted successfully');
      onDatasetChange?.();
    }
  };

  const handleSaveNameChange = async () => {
    if (selectedEntry && editedCourseName.trim()) {
      const updatedEntry: DatasetEntry = {
        ...selectedEntry,
        courseName: editedCourseName,
        filename: `${editedCourseName}_${Date.now()}`,
        location: {
          ...selectedEntry.location,
          name: editedCourseName,
        },
      };

      await updateEntry(updatedEntry);
      const data = await getDatasetAsync();
      setDataset(data);
      setSelectedEntry(updatedEntry);
      toast.success('Course name updated');
      onDatasetChange?.();
    }
  };

  if (isLoading) {
    return (
      <div className="text-center py-8 text-slate-400">
        <div className="size-8 border-2 border-slate-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p>Loading dataset...</p>
      </div>
    );
  }

  if (dataset.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        <ImageIcon className="size-12 mx-auto mb-3 opacity-50" />
        <p>No saved entries yet</p>
        <p className="text-sm mt-1">Capture and save images to see them here</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h4 className="text-slate-300 text-sm font-medium">Saved Entries ({dataset.length})</h4>

      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
        {dataset.map((entry) => (
          <div
            key={entry.id}
            className="bg-slate-700/30 rounded-lg p-2.5 border border-slate-600/30 hover:border-slate-500/50 transition-colors"
          >
            {/* Top section with thumbnail and info */}
            <div className="flex items-start gap-2.5">
              {/* Thumbnail */}
              {entry.images.satellite && (
                <div className="w-14 h-14 rounded overflow-hidden flex-shrink-0 bg-slate-800">
                  <img
                    src={entry.images.satellite}
                    alt={entry.courseName}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}

              {/* Info */}
              <div className="flex-1 min-w-0">
                <h5 className="text-slate-200 font-medium text-xs truncate">
                  {entry.courseName}
                </h5>

                <div className="flex items-center gap-1.5 text-[10px] text-slate-400 mt-0.5">
                  <Calendar className="size-2.5 flex-shrink-0" />
                  <span className="truncate">{new Date(entry.timestamp).toLocaleDateString()}</span>
                </div>

                <div className="flex items-center gap-1.5 text-[10px] text-slate-400 mt-0.5">
                  <MapPin className="size-2.5 flex-shrink-0" />
                  <span className="truncate">
                    {entry.location.lat.toFixed(2)}, {entry.location.lng.toFixed(2)}
                  </span>
                </div>

                {/* Annotations indicator */}
                <div className="flex gap-1 mt-1 flex-wrap">
                  {entry.images.groundTruth && (
                    <span className="px-1.5 py-0.5 bg-purple-500/20 text-purple-400 text-[9px] rounded">
                      Annotated
                    </span>
                  )}
                  {entry.images.mask && (
                    <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 text-[9px] rounded">
                      Predicted
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Actions at bottom */}
            <div className="flex gap-1.5 mt-2.5">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleViewEntry(entry)}
                className="flex-1 h-7 text-[11px] border-slate-600 bg-slate-600 text-white hover:bg-slate-500"
              >
                <Eye className="size-3 mr-1" />
                View
              </Button>

              {entry.images.satellite && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onEditAnnotation(entry)}
                  className="flex-1 h-7 text-[11px] border-slate-600 bg-slate-600 text-white hover:bg-slate-500"
                >
                  <Pencil className="size-3 mr-1" />
                  Edit
                </Button>
              )}

              <Button
                size="sm"
                variant="outline"
                onClick={() => handleDeleteClick(entry)}
                className="h-7 w-7 p-0 border-slate-600 bg-slate-600 text-white hover:bg-red-700"
              >
                <Trash2 className="size-3" />
              </Button>
            </div>
          </div>
        ))}
      </div>

      {/* Preview Dialog */}
      <Dialog open={showPreview} onOpenChange={setShowPreview}>
        <DialogContent className="max-w-4xl bg-slate-800 border-slate-700 text-slate-200">
          <DialogHeader>
            <DialogTitle>{selectedEntry?.courseName}</DialogTitle>
          </DialogHeader>

          {selectedEntry && (
            <div className="space-y-4">
              {/* Images Grid */}
              <div className="grid grid-cols-2 gap-4">
                {/* Satellite Image */}
                {selectedEntry.images.satellite && (
                  <div>
                    <h4 className="text-sm font-medium text-slate-300 mb-2">Satellite Image</h4>
                    <img
                      src={selectedEntry.images.satellite}
                      alt="Satellite"
                      className="w-full rounded border border-slate-600"
                    />
                  </div>
                )}

                {/* Ground Truth */}
                {selectedEntry.images.groundTruth && (
                  <div>
                    <h4 className="text-sm font-medium text-slate-300 mb-2">Ground Truth Annotation</h4>
                    <img
                      src={selectedEntry.images.groundTruth}
                      alt="Ground Truth"
                      className="w-full rounded border border-slate-600"
                    />
                  </div>
                )}

                {/* Model Prediction */}
                {selectedEntry.images.mask && (
                  <div>
                    <h4 className="text-sm font-medium text-slate-300 mb-2">Model Prediction</h4>
                    <img
                      src={selectedEntry.images.mask}
                      alt="Prediction"
                      className="w-full rounded border border-slate-600"
                    />
                  </div>
                )}
              </div>

              {/* Course Name Editor */}
              <div className="space-y-2">
                <Label htmlFor="editCourseName" className="text-slate-300 text-sm">
                  Course Name
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="editCourseName"
                    value={editedCourseName}
                    onChange={(e) => setEditedCourseName(e.target.value)}
                    className="flex-1 bg-slate-700 border-slate-600 text-slate-200"
                  />
                  {editedCourseName !== selectedEntry.courseName && editedCourseName.trim() && (
                    <Button
                      onClick={handleSaveNameChange}
                      className="bg-emerald-600 hover:bg-emerald-700"
                    >
                      <Save className="size-4 mr-2" />
                      Save
                    </Button>
                  )}
                </div>
              </div>

              {/* Metadata */}
              <div className="bg-slate-700/30 rounded p-3 text-sm">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-slate-400">Location:</span>
                    <span className="text-slate-200 ml-2">
                      {selectedEntry.location.lat.toFixed(6)}, {selectedEntry.location.lng.toFixed(6)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400">Timestamp:</span>
                    <span className="text-slate-200 ml-2">
                      {new Date(selectedEntry.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Edit Annotation Button */}
              {selectedEntry.images.satellite && (
                <Button
                  onClick={handleEditAnnotation}
                  className="w-full bg-purple-600 hover:bg-purple-700"
                >
                  <Pencil className="size-4 mr-2" />
                  {selectedEntry.images.groundTruth ? 'Edit Annotation' : 'Add Annotation'}
                </Button>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <DialogContent className="bg-slate-800 border-slate-700 text-slate-200">
          <DialogHeader>
            <DialogTitle>Delete Entry?</DialogTitle>
            <DialogDescription className="text-slate-400">
              Are you sure you want to delete "{entryToDelete?.courseName}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowDeleteConfirm(false)}
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirmDelete}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <Trash2 className="size-4 mr-2" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
