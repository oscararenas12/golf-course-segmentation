'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Undo2, Redo2, Eraser, Save, X, Eye, EyeOff, Sparkles, Paintbrush } from 'lucide-react';
import { toast } from 'sonner';

interface AnnotationCanvasProps {
  imageData: string;
  initialAnnotation?: string;
  onSave: (annotationData: string) => void;
  onClose: () => void;
}

const ANNOTATION_CLASSES = [
  { name: 'Background', color: '#1a1a1a', key: '1' },
  { name: 'Fairway', color: '#2d5016', key: '2' },
  { name: 'Green', color: '#4ade80', key: '3' },
  { name: 'Tee', color: '#ef4444', key: '4' },
  { name: 'Bunker', color: '#fbbf24', key: '5' },
  { name: 'Water', color: '#3b82f6', key: '6' },
];

interface HistoryState {
  imageData: string;
}

export function AnnotationCanvas({ imageData, initialAnnotation, onSave, onClose }: AnnotationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectedClass, setSelectedClass] = useState(ANNOTATION_CLASSES[1]); // Default to Fairway
  const [brushSize, setBrushSize] = useState(20);
  const [isEraser, setIsEraser] = useState(false);
  const [opacity, setOpacity] = useState(60);
  const [showOverlay, setShowOverlay] = useState(true);
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [zoom, setZoom] = useState(100);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [isSamProcessing, setIsSamProcessing] = useState(false);

  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!canvas || !overlayCanvas) return;

    const ctx = canvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');
    if (!ctx || !overlayCtx) return;

    // Load the satellite image
    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;
      overlayCanvas.width = img.width;
      overlayCanvas.height = img.height;

      // Draw background image on main canvas
      ctx.drawImage(img, 0, 0);

      // Initialize overlay with transparent background
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Load initial annotation if provided
      if (initialAnnotation) {
        const annotationImg = new Image();
        annotationImg.onload = () => {
          overlayCtx.drawImage(annotationImg, 0, 0);
          saveToHistory();
        };
        annotationImg.src = initialAnnotation;
      } else {
        // Save initial state
        saveToHistory();
      }
    };
    img.src = imageData;
  }, [imageData, initialAnnotation]);

  // Save state to history
  const saveToHistory = useCallback(() => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const newState: HistoryState = {
      imageData: overlayCanvas.toDataURL(),
    };

    // Remove any states after current index (when undoing then making new changes)
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newState);

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift();
    } else {
      setHistoryIndex(historyIndex + 1);
    }

    setHistory(newHistory);
  }, [history, historyIndex]);

  // Undo
  const handleUndo = useCallback(() => {
    if (historyIndex <= 0) return;

    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    const newIndex = historyIndex - 1;
    const state = history[newIndex];

    const img = new Image();
    img.onload = () => {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.drawImage(img, 0, 0);
    };
    img.src = state.imageData;

    setHistoryIndex(newIndex);
  }, [history, historyIndex]);

  // Redo
  const handleRedo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    const newIndex = historyIndex + 1;
    const state = history[newIndex];

    const img = new Image();
    img.onload = () => {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.drawImage(img, 0, 0);
    };
    img.src = state.imageData;

    setHistoryIndex(newIndex);
  }, [history, historyIndex]);

  // Clear all annotations
  const handleClear = useCallback(() => {
    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCanvas || !overlayCtx) return;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    saveToHistory();
    toast.success('Annotations cleared');
  }, [saveToHistory]);

  // Drawing functions
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    setIsDrawing(true);
    const rect = overlayCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (overlayCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (overlayCanvas.height / rect.height);
    
    draw(x, y);
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveToHistory();
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Update cursor position for brush preview
    setCursorPos({ x: e.clientX, y: e.clientY });

    // Handle panning
    if (isPanning && e.buttons === 4) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      setPanOffset({ x: panOffset.x + dx, y: panOffset.y + dy });
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    if (!isDrawing) return;

    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const rect = overlayCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (overlayCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (overlayCanvas.height / rect.height);

    draw(x, y);
  };

  // Mouse wheel zoom
  const handleWheel = useCallback((e: WheelEvent) => {
    if (!containerRef.current) return;

    e.preventDefault();

    const delta = e.deltaY > 0 ? -10 : 10;
    const newZoom = Math.min(400, Math.max(50, zoom + delta));

    if (newZoom !== zoom) {
      // Calculate zoom toward mouse position
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - rect.width / 2;
      const mouseY = e.clientY - rect.top - rect.height / 2;

      const zoomFactor = newZoom / zoom;
      const newPanX = panOffset.x - mouseX * (zoomFactor - 1);
      const newPanY = panOffset.y - mouseY * (zoomFactor - 1);

      setZoom(newZoom);
      setPanOffset({ x: newPanX, y: newPanY });
    }
  }, [zoom, panOffset]);

  // Handle middle mouse button pan
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 1) { // Middle mouse button
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }
    startDrawing(e);
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 1) {
      setIsPanning(false);
      return;
    }
    stopDrawing();
  };

  const draw = (x: number, y: number) => {
    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas?.getContext('2d');
    if (!overlayCtx) return;

    overlayCtx.beginPath();
    overlayCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);

    if (isEraser) {
      overlayCtx.globalCompositeOperation = 'destination-out';
      overlayCtx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      overlayCtx.globalCompositeOperation = 'source-over';
      overlayCtx.fillStyle = selectedClass.color;
    }

    overlayCtx.fill();
  };

  // Simple Segment Everything - just runs SAM and applies ALL masks as selected class
  const handleSegmentEverything = async () => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    toast.info(`🎯 SAM 2 segmenting everything as ${selectedClass.name}...`);

    try {
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: 'all' }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions.');
        return;
      }

      toast.info(`🔍 Applying ${individualMasks.length} regions as ${selectedClass.name}...`);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing painted pixels to respect
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();
      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0) {
          paintedPixels.add(i / 4);
        }
      }

      const classColor = hexToRgb(selectedClass.color);
      let appliedCount = 0;

      // Apply ALL masks as the selected class
      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
          const pixels = currentData.data;

          let applied = false;
          for (let j = 0; j < maskData.data.length; j += 4) {
            const pixelIndex = j / 4;
            if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
              pixels[j] = classColor.r;
              pixels[j + 1] = classColor.g;
              pixels[j + 2] = classColor.b;
              pixels[j + 3] = 255;
              paintedPixels.add(pixelIndex);
              applied = true;
            }
          }

          if (applied) {
            ctx.putImageData(currentData, 0, 0);
            appliedCount++;
          }
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      saveToHistory();
      toast.success(`✨ Applied ${appliedCount} regions as ${selectedClass.name}. Use eraser to refine.`);

    } catch (error: any) {
      console.error('Segment everything error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Smart Segment All - uses SAM 2 with color-based classification
  const handleSmartSegmentAll = async () => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    toast.info('🎯 SAM 2 analyzing image...');

    try {
      // Get SAM masks
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: 'all' }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions. Try a different image.');
        return;
      }

      toast.info(`🔍 Analyzing ${individualMasks.length} regions...`);

      const baseCtx = baseCanvas.getContext('2d')!;
      const originalImageData = baseCtx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing annotation data to preserve already-painted areas
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();

      // Mark already painted pixels
      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0 || existingData.data[i] > 0 || existingData.data[i + 1] > 0 || existingData.data[i + 2] > 0) {
          paintedPixels.add(i / 4);
        }
      }

      // Classify each mask using color analysis
      const classifiedMasks: { maskUrl: string; className: string; priority: number }[] = [];

      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const colors = analyzeColorsInMask(originalImageData, maskData);

          // Classify based on color
          const className = classifyByColor(colors);
          const priority = getClassPriority(className);
          classifiedMasks.push({ maskUrl, className, priority });
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      // Sort by priority (Water=1 first, Fairway=5 last)
      classifiedMasks.sort((a, b) => a.priority - b.priority);

      const counts: { [key: string]: number } = {};

      // Apply masks in priority order, respecting already-painted pixels
      for (const { maskUrl, className } of classifiedMasks) {
        if (className === 'Background') continue;

        const classInfo = ANNOTATION_CLASSES.find(c => c.name === className);
        if (!classInfo) continue;

        const maskImg = await loadImage(maskUrl);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = overlayCanvas.width;
        tempCanvas.height = overlayCanvas.height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

        const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
        const pixels = currentData.data;
        const classColor = hexToRgb(classInfo.color);

        let applied = false;
        for (let j = 0; j < maskData.data.length; j += 4) {
          const pixelIndex = j / 4;
          // Only paint if mask is active AND pixel isn't already painted
          if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
            pixels[j] = classColor.r;
            pixels[j + 1] = classColor.g;
            pixels[j + 2] = classColor.b;
            pixels[j + 3] = 255;
            paintedPixels.add(pixelIndex);
            applied = true;
          }
        }

        if (applied) {
          ctx.putImageData(currentData, 0, 0);
          counts[className] = (counts[className] || 0) + 1;
        }
      }

      saveToHistory();
      const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ');
      toast.success(`✨ Segmentation complete! Found: ${summary || 'no golf features'}. Use brush to refine.`);

    } catch (error: any) {
      console.error('Segment error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Segment a single class based on color analysis
  const handleSegmentClass = async (targetClass: typeof ANNOTATION_CLASSES[0]) => {
    const overlayCanvas = overlayCanvasRef.current;
    const baseCanvas = canvasRef.current;
    if (!overlayCanvas || !baseCanvas) return;

    setIsSamProcessing(true);
    toast.info(`🎯 SAM 2 finding ${targetClass.name} regions...`);

    try {
      // Step 1: Get all masks from SAM 2
      const response = await fetch('/api/sam-annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData, className: targetClass.name }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'SAM segmentation failed');
      }

      const data = await response.json();
      const individualMasks = data.allMasks || [];

      if (individualMasks.length === 0) {
        toast.warning('SAM 2 found no regions. Try a different image.');
        return;
      }

      toast.info(`🔍 Analyzing ${individualMasks.length} regions for ${targetClass.name}...`);

      // Get original image data for color analysis
      const baseCtx = baseCanvas.getContext('2d')!;
      const originalImageData = baseCtx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);

      const ctx = overlayCanvas.getContext('2d')!;

      // Get existing annotation data to preserve already-painted areas
      const existingData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const paintedPixels = new Set<number>();

      // Mark already painted pixels AND check if this class was already used
      const targetColor = hexToRgb(targetClass.color);
      let sameClassPixelCount = 0;

      for (let i = 0; i < existingData.data.length; i += 4) {
        if (existingData.data[i + 3] > 0 || existingData.data[i] > 0 || existingData.data[i + 1] > 0 || existingData.data[i + 2] > 0) {
          paintedPixels.add(i / 4);
          // Check if this pixel is already our target class color
          if (Math.abs(existingData.data[i] - targetColor.r) < 5 &&
              Math.abs(existingData.data[i + 1] - targetColor.g) < 5 &&
              Math.abs(existingData.data[i + 2] - targetColor.b) < 5) {
            sameClassPixelCount++;
          }
        }
      }

      // If we already have some of this class, use more lenient detection (second pass)
      const isSecondPass = sameClassPixelCount > 1000;
      if (isSecondPass) {
        toast.info(`🔄 Second pass for ${targetClass.name} - using broader detection...`);
      }

      let appliedCount = 0;
      let analyzedCount = 0;

      // Process ALL masks for thorough analysis
      for (let i = 0; i < individualMasks.length; i++) {
        const maskUrl = individualMasks[i];
        if (!maskUrl) continue;

        try {
          const maskImg = await loadImage(maskUrl);

          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = overlayCanvas.width;
          tempCanvas.height = overlayCanvas.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

          const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const colors = analyzeColorsInMask(originalImageData, maskData);
          const detectedClass = classifyByColor(colors, targetClass.name, isSecondPass);

          analyzedCount++;

          // Only apply if it matches the target class
          if (detectedClass !== targetClass.name) continue;

          const currentData = ctx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
          const pixels = currentData.data;
          const classColor = hexToRgb(targetClass.color);

          let applied = false;
          for (let j = 0; j < maskData.data.length; j += 4) {
            const pixelIndex = j / 4;
            // Only paint if mask is active AND pixel isn't already painted
            if ((maskData.data[j] > 128 || maskData.data[j + 1] > 128 || maskData.data[j + 2] > 128) && !paintedPixels.has(pixelIndex)) {
              pixels[j] = classColor.r;
              pixels[j + 1] = classColor.g;
              pixels[j + 2] = classColor.b;
              pixels[j + 3] = 255;
              paintedPixels.add(pixelIndex); // Mark as painted
              applied = true;
            }
          }

          if (applied) {
            ctx.putImageData(currentData, 0, 0);
            appliedCount++;
          }
        } catch (e) {
          console.warn(`Failed to process mask ${i}:`, e);
        }
      }

      saveToHistory();

      if (appliedCount > 0) {
        const passInfo = isSecondPass ? ' (broader search)' : '';
        toast.success(`✨ Applied ${appliedCount} ${targetClass.name} region${appliedCount > 1 ? 's' : ''}${passInfo}. Use brush to refine.`);
      } else {
        const suggestion = isSecondPass
          ? 'No more found. Try manual annotation.'
          : 'Try clicking again for broader search, or use manual annotation.';
        toast.warning(`No ${targetClass.name} found. ${suggestion}`);
      }

    } catch (error: any) {
      console.error('Segment class error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsSamProcessing(false);
    }
  };

  // Helper: Load image as promise
  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = url;
    });
  };

  // Helper: Analyze colors in original image where mask is active
  const analyzeColorsInMask = (
    originalData: ImageData,
    maskData: ImageData
  ): { avgR: number; avgG: number; avgB: number; pixelCount: number } => {
    let totalR = 0, totalG = 0, totalB = 0, count = 0;

    for (let i = 0; i < maskData.data.length; i += 4) {
      if (maskData.data[i] > 128) {
        totalR += originalData.data[i];
        totalG += originalData.data[i + 1];
        totalB += originalData.data[i + 2];
        count++;
      }
    }

    return {
      avgR: count > 0 ? totalR / count : 0,
      avgG: count > 0 ? totalG / count : 0,
      avgB: count > 0 ? totalB / count : 0,
      pixelCount: count
    };
  };

  // Helper: Classify region by average color with target class focus (global fallback)
  const classifyByColor = (
    colors: { avgR: number; avgG: number; avgB: number; pixelCount: number },
    targetClass?: string,
    isSecondPass?: boolean
  ): string => {
    const { avgR, avgG, avgB, pixelCount } = colors;

    // On second pass, be more lenient with size requirements
    const minPixels = isSecondPass ? 150 : (targetClass === 'Tee' ? 200 : 300);
    if (pixelCount < minPixels) return 'Background';

    // Second pass multiplier - boost target class score more aggressively
    const secondPassBoost = isSecondPass ? 2.0 : 1.0;

    // Calculate useful color metrics
    const brightness = (avgR + avgG + avgB) / 3;
    const greenDominance = avgG - Math.max(avgR, avgB);
    const blueDominance = avgB - Math.max(avgR, avgG);
    const saturation = Math.max(avgR, avgG, avgB) - Math.min(avgR, avgG, avgB);

    // Score each class based on how well colors match
    const scores: { [key: string]: number } = {
      'Water': 0,
      'Bunker': 0,
      'Green': 0,
      'Tee': 0,
      'Fairway': 0,
    };

    // WATER: Blue tones - MUST have blue dominant, NOT green dominant (that's trees!)
    // Key: Water has blue >= green, Trees have green > blue
    if (avgB > avgR && avgB >= avgG - 5) {
      scores['Water'] += 30;
      if (avgB > avgG) scores['Water'] += 25; // Blue clearly dominant = likely water
      if (avgB > 80) scores['Water'] += 10;
      // Penalize if green is dominant - that's trees, not water!
      if (avgG > avgB + 15) scores['Water'] -= 50; // Green dominant = trees
      if (avgG > avgR && avgG > avgB && brightness < 80) scores['Water'] -= 60; // Dark green = definitely trees
    }
    // Extra penalty for dark green areas being classified as water
    if (avgG > avgB + 10 && avgG > avgR && brightness < 90) {
      scores['Water'] -= 40; // This is trees, not water
    }

    // BUNKER: Sandy/tan colors - high R and G, lower B, not too bright
    if (avgR > 140 && avgG > 120 && avgB < avgG + 20) {
      scores['Bunker'] += 25;
      if (avgR > avgB + 30) scores['Bunker'] += 20;
      if (avgG > avgB + 20) scores['Bunker'] += 15;
      if (brightness > 140 && brightness < 220) scores['Bunker'] += 15;
      // Sandy look: R and G close together, both higher than B
      if (Math.abs(avgR - avgG) < 40 && avgR > avgB) scores['Bunker'] += 15;
    }

    // GREEN (putting green): Small, bright green areas
    // KEY DISTINCTION: Greens are SMALL (typically under 5000 pixels)
    if (greenDominance > 10 && avgG > 100 && brightness > 95) {
      // Base score for green-ish color
      scores['Green'] += 15;
      if (avgG > 120) scores['Green'] += 10;
      if (greenDominance > 20) scores['Green'] += 10;
      if (saturation > 40) scores['Green'] += 10;

      // SIZE IS KEY: Greens are small!
      if (pixelCount < 3000) scores['Green'] += 40;  // Very likely a green
      else if (pixelCount < 6000) scores['Green'] += 25;
      else if (pixelCount < 10000) scores['Green'] += 5;
      else scores['Green'] -= 30; // Too big = probably fairway, not green
    }

    // TEE: Very small bright green areas
    if (greenDominance > 8 && avgG > 95 && brightness > 90) {
      scores['Tee'] += 10;
      // Tees are VERY small
      if (pixelCount < 1500) scores['Tee'] += 45;
      else if (pixelCount < 2500) scores['Tee'] += 30;
      else if (pixelCount < 4000) scores['Tee'] += 10;
      else scores['Tee'] -= 25; // Too big for a tee
    }

    // FAIRWAY: Green areas that aren't small enough to be greens/tees
    // More lenient on size, focus on "green but not tiny"
    if (avgG > avgR && avgG > avgB && brightness > 75 && brightness < 150) {
      scores['Fairway'] += 20; // Base score for any greenish area
      if (avgG > 80) scores['Fairway'] += 10;
      if (greenDominance > 3) scores['Fairway'] += 10;

      // Size bonuses (but not as strict)
      if (pixelCount > 10000) scores['Fairway'] += 25; // Large = likely fairway
      else if (pixelCount > 5000) scores['Fairway'] += 15;
      else if (pixelCount > 3000) scores['Fairway'] += 10;
      // Small penalty but not too harsh
      else if (pixelCount < 2000) scores['Fairway'] -= 10;

      // Penalize if too dark (likely trees)
      if (brightness < 70) scores['Fairway'] -= 35;
    }

    // TREES/ROUGH penalty: Dark green areas should be Background
    if (brightness < 65 && avgG > avgR) {
      scores['Fairway'] -= 40;
      scores['Green'] -= 25;
      scores['Tee'] -= 25;
    }

    // If targeting a specific class, boost its score to be more inclusive
    // On second pass, boost even more aggressively to find borderline matches
    if (targetClass && scores[targetClass] !== undefined) {
      const boost = isSecondPass ? 2.5 : 1.5;
      scores[targetClass] *= boost;
    }

    // On second pass, also lower the minimum threshold to catch more
    const minThreshold = isSecondPass ? 15 : 25;

    // Find the highest scoring class
    let bestClass = 'Background';
    let bestScore = minThreshold; // Minimum threshold to be classified

    for (const [className, score] of Object.entries(scores)) {
      if (score > bestScore) {
        bestScore = score;
        bestClass = className;
      }
    }

    return bestClass;
  };

  // Helper: Check if color is compatible with a class (sanity check for GPT suggestions)
  const isColorCompatible = (
    colors: { avgR: number; avgG: number; avgB: number; pixelCount: number },
    className: string
  ): boolean => {
    const { avgR, avgG, avgB } = colors;
    const brightness = (avgR + avgG + avgB) / 3;

    switch (className) {
      case 'Water':
        // Water MUST have blue dominant or near-dominant, NOT green dominant (that's trees!)
        // Reject if green is clearly dominant over blue
        if (avgG > avgB + 15) return false; // Green dominant = trees, not water
        if (avgG > avgR && avgG > avgB && brightness < 85) return false; // Dark green = trees
        return avgB > avgR - 20 && avgB >= avgG - 10; // Blue should be significant
      case 'Bunker':
        // Bunker should be tan/sandy (not green or blue dominant)
        return avgR > 100 && avgG > 90 && avgB < avgG + 40;
      case 'Green':
      case 'Tee':
      case 'Fairway':
        // These MUST be green - G must be dominant, not sandy/tan
        // Reject if it looks like a bunker (R and G high, similar values, B lower)
        const isSandy = avgR > 140 && avgG > 120 && Math.abs(avgR - avgG) < 50 && avgB < avgG;
        if (isSandy) return false;
        // Must have green as the dominant or near-dominant color
        return avgG > avgR - 10 && avgG > avgB && brightness > 60;
      default:
        return true;
    }
  };

  // Helper: Priority for layering (lower = applied first, won't be overwritten)
  const getClassPriority = (className: string): number => {
    const priorities: { [key: string]: number } = {
      'Water': 1,
      'Bunker': 2,
      'Green': 3,
      'Tee': 4,
      'Fairway': 5,
      'Background': 99
    };
    return priorities[className] || 99;
  };

  // Helper: Convert hex color to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  };

  // Handle save
  const handleSave = () => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    // Create a composite canvas with background fill
    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = overlayCanvas.width;
    compositeCanvas.height = overlayCanvas.height;
    const ctx = compositeCanvas.getContext('2d')!;

    // Fill entire canvas with background color (everything unannotated = background)
    ctx.fillStyle = ANNOTATION_CLASSES[0].color; // Background color
    ctx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);

    // Draw the user's annotations on top
    ctx.drawImage(overlayCanvas, 0, 0);

    // Export the composite
    const annotationData = compositeCanvas.toDataURL('image/png');
    onSave(annotationData);
    toast.success('Annotation saved! Unannotated areas = Background');
  };

  // Mouse wheel zoom listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Number keys for class selection
      const num = parseInt(e.key);
      if (num >= 1 && num <= 6) {
        setSelectedClass(ANNOTATION_CLASSES[num - 1]);
        setIsEraser(false);
        return;
      }

      // Zoom shortcuts
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        setZoom(prev => Math.min(400, prev + 25));
        return;
      }
      if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        setZoom(prev => Math.max(50, prev - 25));
        return;
      }
      if (e.key === '0' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setZoom(100);
        setPanOffset({ x: 0, y: 0 });
        return;
      }

      // Shortcuts
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z') {
          e.preventDefault();
          if (e.shiftKey) {
            handleRedo();
          } else {
            handleUndo();
          }
        }
      }

      if (e.key === 'e' || e.key === 'E') {
        setIsEraser(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  return (
    <div className="fixed inset-0 bg-black/95 z-50 flex flex-col">
      {/* Header */}
      <div className="bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-white text-xl">Annotation Tool</h2>
          <p className="text-slate-400 text-sm">Draw to annotate golf course features</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            className="bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-600"
          >
            <Save className="size-4 mr-2" />
            Save Annotation
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="size-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden relative">
        {/* Sidebar Controls */}
        <div className="w-80 bg-slate-900 border-r border-slate-700 p-6 overflow-y-auto">
          {/* Per-Class Segment Buttons */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">AI Segment by Class</Label>
            <p className="text-slate-400 text-xs mb-3">
              Click twice for broader search. Won't overwrite existing.
            </p>
            <div className="grid grid-cols-2 gap-2">
              {ANNOTATION_CLASSES.filter(cls => cls.name !== 'Background').map((cls) => (
                <Button
                  key={cls.name}
                  onClick={() => handleSegmentClass(cls)}
                  disabled={isSamProcessing}
                  className="h-10 text-xs font-medium disabled:opacity-50"
                  style={{
                    backgroundColor: cls.color,
                    color: cls.name === 'Bunker' ? '#000' : '#fff',
                  }}
                >
                  {isSamProcessing ? (
                    <div className="size-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <>
                      <Sparkles className="size-3 mr-1" />
                      {cls.name}
                    </>
                  )}
                </Button>
              ))}
            </div>
            {/* Segment All button */}
            <Button
              onClick={handleSmartSegmentAll}
              disabled={isSamProcessing}
              className="w-full h-12 mt-3 text-sm font-medium disabled:opacity-50 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              {isSamProcessing ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing all classes...
                </>
              ) : (
                <>
                  <Sparkles className="size-4 mr-2" />
                  Segment All Classes
                </>
              )}
            </Button>
            <p className="text-slate-400 text-xs mt-2 text-center">
              GPT-4 + SAM 2 segments all features with smart priority
            </p>

            {/* Segment Everything button */}
            <Button
              onClick={handleSegmentEverything}
              disabled={isSamProcessing}
              className="w-full h-10 mt-3 text-sm font-medium disabled:opacity-50 bg-slate-600 hover:bg-slate-500"
            >
              {isSamProcessing ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Paintbrush className="size-4 mr-2" />
                  Segment Everything
                </>
              )}
            </Button>
            <p className="text-slate-500 text-xs mt-1 text-center">
              SAM 2 segments all regions (uses selected class color)
            </p>
          </div>

          {/* Class Selection */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">Select Class</Label>
            <div className="bg-blue-950/30 border border-blue-600/30 rounded p-2 mb-3">
              <p className="text-xs text-blue-300">
                💡 Tip: Only paint features. Unannotated areas automatically become Background!
              </p>
            </div>
            <div className="space-y-2">
              {ANNOTATION_CLASSES.map((cls, idx) => (
                <button
                  key={cls.name}
                  onClick={() => {
                    setSelectedClass(cls);
                    setIsEraser(false);
                  }}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                    selectedClass.name === cls.name && !isEraser
                      ? 'bg-slate-700 ring-2 ring-emerald-500'
                      : idx === 0
                      ? 'bg-slate-800/50 hover:bg-slate-700/50 opacity-60'
                      : 'bg-slate-800 hover:bg-slate-700'
                  }`}
                >
                  <div
                    className="size-6 rounded border-2 border-slate-600"
                    style={{ backgroundColor: cls.color }}
                  />
                  <div className="flex-1 text-left">
                    <span className="text-white">{cls.name}</span>
                    {idx === 0 && (
                      <span className="text-xs text-slate-500 block">Auto-filled</span>
                    )}
                  </div>
                  <span className="text-slate-500 text-sm">{cls.key}</span>
                </button>
              ))}
              
              {/* Eraser */}
              <button
                onClick={() => setIsEraser(true)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                  isEraser
                    ? 'bg-slate-700 ring-2 ring-red-500'
                    : 'bg-slate-800 hover:bg-slate-700'
                }`}
              >
                <div className="size-6 rounded border-2 border-slate-600 bg-slate-900 flex items-center justify-center">
                  <Eraser className="size-4 text-slate-400" />
                </div>
                <span className="text-white flex-1 text-left">Eraser</span>
                <span className="text-slate-500 text-sm">E</span>
              </button>
            </div>
          </div>

          {/* Brush Size */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Brush Size: {brushSize}px
            </Label>
            <Slider
              value={[brushSize]}
              onValueChange={(values) => setBrushSize(values[0])}
              min={5}
              max={100}
              step={5}
              className="mb-2"
            />
          </div>

          {/* Overlay Opacity */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Overlay Opacity: {opacity}%
            </Label>
            <Slider
              value={[opacity]}
              onValueChange={(values) => setOpacity(values[0])}
              min={0}
              max={100}
              step={5}
            />
          </div>

          {/* Zoom */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">
              Zoom: {zoom}%
            </Label>
            <Slider
              value={[zoom]}
              onValueChange={(values) => setZoom(values[0])}
              min={50}
              max={400}
              step={10}
            />
            <div className="flex gap-2 mt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setZoom(Math.max(50, zoom - 25))}
                className="flex-1"
              >
                -
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setZoom(100);
                  setPanOffset({ x: 0, y: 0 });
                }}
                className="flex-1"
              >
                Reset
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setZoom(Math.min(400, zoom + 25))}
                className="flex-1"
              >
                +
              </Button>
            </div>
          </div>

          {/* Tools */}
          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowOverlay(!showOverlay)}
              className="w-full"
            >
              {showOverlay ? (
                <>
                  <EyeOff className="size-4 mr-2" />
                  Hide Overlay
                </>
              ) : (
                <>
                  <Eye className="size-4 mr-2" />
                  Show Overlay
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleUndo}
              disabled={historyIndex <= 0}
              className="w-full"
            >
              <Undo2 className="size-4 mr-2" />
              Undo (Ctrl+Z)
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleRedo}
              disabled={historyIndex >= history.length - 1}
              className="w-full"
            >
              <Redo2 className="size-4 mr-2" />
              Redo (Ctrl+Shift+Z)
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleClear}
              className="w-full text-red-400 border-red-400 hover:bg-red-950"
            >
              <X className="size-4 mr-2" />
              Clear All
            </Button>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="mt-6 pt-6 border-t border-slate-700">
            <Label className="text-white mb-3 block">Controls</Label>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="flex justify-between">
                <span>1-6</span>
                <span>Select class</span>
              </div>
              <div className="flex justify-between">
                <span>E</span>
                <span>Toggle eraser</span>
              </div>
              <div className="flex justify-between">
                <span>Scroll</span>
                <span>Zoom</span>
              </div>
              <div className="flex justify-between">
                <span>Middle Mouse</span>
                <span>Pan</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+0</span>
                <span>Reset view</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+Z</span>
                <span>Undo</span>
              </div>
              <div className="flex justify-between">
                <span>Ctrl+Shift+Z</span>
                <span>Redo</span>
              </div>
            </div>
          </div>
        </div>

        {/* Canvas Area */}
        <div
          ref={containerRef}
          className="flex-1 flex items-center justify-center overflow-hidden bg-slate-950 relative"
        >
          <div
            className="relative transition-transform duration-100 ease-out"
            style={{
              cursor: isPanning ? 'grabbing' : 'none',
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom / 100})`,
              transformOrigin: 'center',
            }}
          >
            {/* Base image canvas */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0"
              style={{ imageRendering: 'auto' }}
            />

            {/* Annotation overlay canvas */}
            <canvas
              ref={overlayCanvasRef}
              className="relative"
              style={{
                opacity: showOverlay ? opacity / 100 : 0,
                imageRendering: 'auto',
                cursor: 'none',
              }}
              onMouseDown={handleCanvasMouseDown}
              onMouseUp={handleCanvasMouseUp}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => {
                stopDrawing();
                setIsHovering(false);
                setIsPanning(false);
              }}
              onMouseEnter={() => setIsHovering(true)}
            />
          </div>

          {/* Brush cursor indicator */}
          {isHovering && !isPanning && (
            <div
              className="fixed pointer-events-none rounded-full border-2 border-white mix-blend-difference transition-all duration-75"
              style={{
                width: brushSize * (zoom / 100),
                height: brushSize * (zoom / 100),
                transform: 'translate(-50%, -50%)',
                left: cursorPos.x,
                top: cursorPos.y,
                backgroundColor: isEraser ? 'rgba(255,0,0,0.2)' : `${selectedClass.color}40`,
              }}
            />
          )}

          {/* Zoom indicator */}
          {zoom !== 100 && (
            <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded-lg text-sm">
              {zoom}%
            </div>
          )}

          {/* Pan hint */}
          {zoom > 100 && !isPanning && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-slate-300 px-3 py-1 rounded-lg text-xs">
              Scroll to zoom • Middle mouse to pan
            </div>
          )}
        </div>

      </div>

      {/* Status Bar */}
      <div className="bg-slate-900 border-t border-slate-700 px-6 py-2 flex items-center justify-between text-sm">
        <div className="text-slate-400">
          {isEraser ? (
            <span className="text-red-400">Eraser Mode</span>
          ) : (
            <span>
              Drawing: <span style={{ color: selectedClass.color }}>{selectedClass.name}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div className="text-slate-500">
            Zoom: {zoom}%
          </div>
          <div className="text-slate-500">
            Canvas: {canvasRef.current?.width || 0} × {canvasRef.current?.height || 0} px
          </div>
        </div>
      </div>
    </div>
  );
}
