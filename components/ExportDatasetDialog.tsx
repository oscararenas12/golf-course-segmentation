'use client';

import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { RadioGroup, RadioGroupItem } from './ui/radio-group';
import { Checkbox } from './ui/checkbox';
import { Download, Folder, Cloud } from 'lucide-react';
import { getDatasetAsync, formatBytes, calculateDatasetSizeSync } from '../utils/datasetStorage';
import { DatasetEntry } from '../types/dataset';
import { toast } from 'sonner';
import JSZip from 'jszip';

interface ExportDatasetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ExportDatasetDialog({ open, onOpenChange }: ExportDatasetDialogProps) {
  const [datasetName, setDatasetName] = useState('my_golf_courses_2024');
  const [exportFormat, setExportFormat] = useState<'zip' | 'folder' | 'cloud'>('zip');
  const [includeOptions, setIncludeOptions] = useState({
    images: true,
    masks: true,
    metadata: true,
    readme: true,
  });
  const [isExporting, setIsExporting] = useState(false);
  const [dataset, setDataset] = useState<DatasetEntry[]>([]);
  const [datasetSize, setDatasetSize] = useState('0 Bytes');

  // Load dataset when dialog opens
  useEffect(() => {
    if (open) {
      const loadDataset = async () => {
        const data = await getDatasetAsync();
        setDataset(data);
        setDatasetSize(formatBytes(calculateDatasetSizeSync()));
      };
      loadDataset();
    }
  }, [open]);

  const handleExport = async () => {
    setIsExporting(true);
    
    try {
      const zip = new JSZip();
      
      // Create folder structure
      const imagesFolder = zip.folder('images');
      const masksFolder = zip.folder('masks');
      const metadataFolder = zip.folder('metadata');

      // Add dataset entries
      dataset.forEach((entry, index) => {
        const baseName = `course_${String(index + 1).padStart(3, '0')}`;
        
        // Add satellite image
        if (includeOptions.images && entry.images.satellite) {
          const imageData = entry.images.satellite.split(',')[1]; // Remove data URL prefix
          imagesFolder?.file(`${baseName}.jpg`, imageData, { base64: true });
        }
        
        // Add model prediction mask (if exists)
        if (includeOptions.masks && entry.images.mask) {
          const maskData = entry.images.mask.split(',')[1];
          masksFolder?.file(`${baseName}_prediction.png`, maskData, { base64: true });
        }

        // Add ground truth annotation (if exists)
        if (includeOptions.masks && entry.images.groundTruth) {
          const gtData = entry.images.groundTruth.split(',')[1];
          masksFolder?.file(`${baseName}_ground_truth.png`, gtData, { base64: true });
        }
        
        // Add metadata
        if (includeOptions.metadata) {
          const metadata = {
            filename: `${baseName}.jpg`,
            timestamp: entry.timestamp,
            location: entry.location,
            capture_box: entry.captureBox,
            segmentation: entry.segmentation,
            user_notes: entry.userNotes || '',
          };
          metadataFolder?.file(`${baseName}_meta.json`, JSON.stringify(metadata, null, 2));
        }
      });

      // Add dataset info
      if (includeOptions.metadata) {
        const datasetInfo = {
          name: datasetName,
          created_at: new Date().toISOString(),
          total_images: dataset.length,
          format: 'Golf Course Segmentation Dataset',
          classes: [
            'background',
            'fairway',
            'green',
            'tee',
            'bunker',
            'water',
          ],
          capture_box_size: {
            width: 1664,
            height: 1024,
          },
        };
        zip.file('dataset_info.json', JSON.stringify(datasetInfo, null, 2));
      }

      // Add README
      if (includeOptions.readme) {
        const readme = `# ${datasetName}

Golf Course Segmentation Dataset

## Dataset Information

- **Total Images**: ${dataset.length}
- **Image Size**: 1664 × 1024 pixels
- **Format**: JPEG (images), PNG (masks)
- **Created**: ${new Date().toLocaleDateString()}

## Directory Structure

\`\`\`
${datasetName}/
├── images/                    # Satellite imagery
│   └── course_001.jpg
├── masks/                     # Segmentation masks
│   ├── course_001_ground_truth.png    # Manual annotations
│   └── course_001_prediction.png      # Model predictions (if available)
├── metadata/                  # JSON metadata for each image
│   └── course_001_meta.json
├── dataset_info.json
└── README.md
\`\`\`

## Mask Types

- **ground_truth.png**: Manual annotations created with the annotation tool
  - Use these as training labels for supervised learning
  - Most accurate, but requires manual effort

- **prediction.png**: Model predictions from ResNet50 U-Net
  - Use these as pseudo-labels or for model evaluation
  - Generated automatically, less accurate

## Classes

0. Background (RGB: 26, 26, 26)
1. Fairway (RGB: 45, 80, 22)
2. Green (RGB: 74, 222, 128)
3. Tee (RGB: 239, 68, 68)
4. Bunker (RGB: 251, 191, 36)
5. Water (RGB: 59, 130, 246)

## Usage

This dataset can be used for training semantic segmentation models for golf course feature detection.

### Training Split Suggestion

- Training: 80% (${Math.floor(dataset.length * 0.8)} images)
- Validation: 20% (${Math.ceil(dataset.length * 0.2)} images)

### Loading Ground Truth Masks

\`\`\`python
import numpy as np
from PIL import Image

# Load ground truth annotation
mask = Image.open('masks/course_001_ground_truth.png')
mask_array = np.array(mask)

# Extract class labels by color matching
# Or convert RGB to class indices for training
\`\`\`

## Notes

Generated using Golf Course Segmentation Tool
`;
        zip.file('README.md', readme);
      }

      // Generate and download ZIP
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetName}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success('✅ Dataset exported successfully!');
      onOpenChange(false);
      
    } catch (error) {
      toast.error('❌ Failed to export dataset');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px] bg-slate-800 border-slate-700 text-slate-200">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            📦 Export Dataset
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            Export {dataset.length} images from this session
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Export Format */}
          <div>
            <Label className="text-slate-300 mb-3 block">Format:</Label>
            <RadioGroup value={exportFormat} onValueChange={(value: any) => setExportFormat(value)}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="zip" id="zip" />
                <Label htmlFor="zip" className="text-slate-300 cursor-pointer flex items-center gap-2">
                  <Download className="size-4" />
                  ZIP Archive (download)
                </Label>
              </div>
              <div className="flex items-center space-x-2 opacity-50">
                <RadioGroupItem value="folder" id="folder" disabled />
                <Label htmlFor="folder" className="text-slate-400 cursor-not-allowed flex items-center gap-2">
                  <Folder className="size-4" />
                  Folder Structure (coming soon)
                </Label>
              </div>
              <div className="flex items-center space-x-2 opacity-50">
                <RadioGroupItem value="cloud" id="cloud" disabled />
                <Label htmlFor="cloud" className="text-slate-400 cursor-not-allowed flex items-center gap-2">
                  <Cloud className="size-4" />
                  Cloud Storage (coming soon)
                </Label>
              </div>
            </RadioGroup>
          </div>

          {/* Structure Preview */}
          <div className="bg-slate-900 rounded p-3 text-xs font-mono text-slate-300">
            <div>{datasetName}/</div>
            <div className="ml-4">├── images/</div>
            <div className="ml-8">│   ├── course_001.jpg</div>
            <div className="ml-8">│   └── ...</div>
            <div className="ml-4">├── masks/</div>
            <div className="ml-8">│   ├── course_001_ground_truth.png</div>
            <div className="ml-8">│   ├── course_001_prediction.png</div>
            <div className="ml-8">│   └── ...</div>
            <div className="ml-4">├── metadata/</div>
            <div className="ml-8">│   └── course_001_meta.json</div>
            <div className="ml-4">├── dataset_info.json</div>
            <div className="ml-4">└── README.md</div>
          </div>

          {/* Include Options */}
          <div>
            <Label className="text-slate-300 mb-2 block">Include:</Label>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeImages"
                  checked={includeOptions.images}
                  onCheckedChange={(checked) =>
                    setIncludeOptions({ ...includeOptions, images: checked as boolean })
                  }
                />
                <label htmlFor="includeImages" className="text-slate-300 text-sm cursor-pointer">
                  Images (JPEG)
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeMasks"
                  checked={includeOptions.masks}
                  onCheckedChange={(checked) =>
                    setIncludeOptions({ ...includeOptions, masks: checked as boolean })
                  }
                />
                <label htmlFor="includeMasks" className="text-slate-300 text-sm cursor-pointer">
                  Masks (PNG)
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeMetadata"
                  checked={includeOptions.metadata}
                  onCheckedChange={(checked) =>
                    setIncludeOptions({ ...includeOptions, metadata: checked as boolean })
                  }
                />
                <label htmlFor="includeMetadata" className="text-slate-300 text-sm cursor-pointer">
                  Metadata (JSON)
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeReadme"
                  checked={includeOptions.readme}
                  onCheckedChange={(checked) =>
                    setIncludeOptions({ ...includeOptions, readme: checked as boolean })
                  }
                />
                <label htmlFor="includeReadme" className="text-slate-300 text-sm cursor-pointer">
                  README.md
                </label>
              </div>
            </div>
          </div>

          {/* Dataset Name */}
          <div>
            <Label htmlFor="datasetName" className="text-slate-300 mb-2 block">
              Dataset Name:
            </Label>
            <Input
              id="datasetName"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className="bg-slate-700 border-slate-600 text-slate-200"
            />
          </div>

          {/* Size Info */}
          <div className="bg-slate-700/30 rounded p-2 text-sm text-slate-400">
            Total size: {datasetSize}
          </div>

          {/* Export Button */}
          <Button
            onClick={handleExport}
            disabled={isExporting || dataset.length === 0}
            className="w-full bg-emerald-500 hover:bg-emerald-600"
          >
            {isExporting ? (
              <>
                <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="size-4 mr-2" />
                Export & Download
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}