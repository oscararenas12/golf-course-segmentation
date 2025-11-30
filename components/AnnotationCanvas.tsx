'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Undo2, Redo2, Eraser, Save, X, Eye, EyeOff, Sparkles } from 'lucide-react';
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

// Map from API class names to our annotation class colors
const API_CLASS_COLORS: { [key: string]: string } = {
  'Background': '#1a1a1a',
  'Fairway': '#2d5016',
  'Green': '#4ade80',
  'Tee': '#ef4444',
  'Bunker': '#fbbf24',
  'Water': '#3b82f6',
};

interface HistoryState {
  imageData: string;
}

const SEGMENTATION_API_URL = process.env.NEXT_PUBLIC_SEGMENTATION_API_URL || 'https://elo0oo0-golf-segmentation-api.hf.space';

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
  const [isProcessing, setIsProcessing] = useState(false);

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

  // Auto-segment using HuggingFace U-Net model
  const handleAutoSegment = async () => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    setIsProcessing(true);
    toast.info('Running U-Net segmentation model...');

    try {
      const response = await fetch(`${SEGMENTATION_API_URL}/segment-base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageData }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Segmentation failed');
      }

      const data = await response.json();

      if (!data.success || !data.overlayData) {
        throw new Error('Invalid response from segmentation API');
      }

      // Load the segmentation result
      const segmentationImg = new Image();
      segmentationImg.crossOrigin = 'anonymous';

      segmentationImg.onload = () => {
        const ctx = overlayCanvas.getContext('2d')!;

        // Clear existing and draw segmentation result
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        ctx.drawImage(segmentationImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

        saveToHistory();

        // Show statistics
        const stats = data.statistics;
        const summary = Object.entries(stats)
          .filter(([name, value]: [string, any]) => name !== 'Background' && value.percentage > 0.5)
          .map(([name, value]: [string, any]) => `${name}: ${value.percentage.toFixed(1)}%`)
          .join(', ');

        toast.success(`Segmentation complete! ${summary || 'No golf features detected'}`);
      };

      segmentationImg.onerror = () => {
        throw new Error('Failed to load segmentation result');
      };

      segmentationImg.src = data.overlayData;

    } catch (error: any) {
      console.error('Auto-segment error:', error);
      toast.error(`Segmentation failed: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
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
    ctx.fillStyle = ANNOTATION_CLASSES[0].color;
    ctx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);

    // Draw the user's annotations on top
    ctx.drawImage(overlayCanvas, 0, 0);

    // Export the composite
    const annotationData = compositeCanvas.toDataURL('image/png');
    onSave(annotationData);
    toast.success('Annotation saved!');
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
          {/* Auto Segment Button */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">AI Auto-Segment</Label>
            <Button
              onClick={handleAutoSegment}
              disabled={isProcessing}
              className="w-full h-12 text-sm font-medium disabled:opacity-50 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
            >
              {isProcessing ? (
                <>
                  <div className="size-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles className="size-4 mr-2" />
                  Auto-Segment with U-Net
                </>
              )}
            </Button>
            <p className="text-slate-400 text-xs mt-2 text-center">
              Uses trained U-Net model to detect golf features
            </p>
          </div>

          {/* Class Selection */}
          <div className="mb-6">
            <Label className="text-white mb-3 block">Select Class</Label>
            <div className="bg-blue-950/30 border border-blue-600/30 rounded p-2 mb-3">
              <p className="text-xs text-blue-300">
                Tip: Use Auto-Segment first, then refine with brush
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
                <span>Ctrl+Z</span>
                <span>Undo</span>
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
            Canvas: {canvasRef.current?.width || 0} x {canvasRef.current?.height || 0} px
          </div>
        </div>
      </div>
    </div>
  );
}
