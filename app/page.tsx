'use client';

import Link from 'next/link';
import { Database, Target, Tag, ArrowRight, ExternalLink } from 'lucide-react';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { Toaster } from '@/components/ui/sonner';

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950">
      <Header />

      <div className="flex-1 py-12">
        <div className="max-w-6xl w-full px-8 mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="flex items-center justify-center gap-3 mb-4">
              <span className="text-6xl">⛳</span>
            </div>
            <h1 className="text-white text-4xl mb-3">
              Golf Course Segmentation
            </h1>
            <p className="text-slate-400 text-lg">
              Choose your workflow to get started
            </p>
          </div>

          {/* Workflow Options - 3 columns */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* Dataset Creation */}
            <Link href="/dataset">
              <div className="group relative bg-slate-800 border-2 border-slate-700 rounded-2xl p-6 text-left transition-all hover:border-blue-500 hover:shadow-xl hover:shadow-blue-500/20 hover:-translate-y-1 h-full cursor-pointer">
                <div className="absolute top-4 left-4 px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded-full border border-blue-500/30">
                  ML Training
                </div>

                <div className="absolute top-4 right-4 size-10 bg-blue-500/10 rounded-full flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                  <Database className="size-5 text-blue-400" />
                </div>

                <div className="mb-4 mt-8">
                  <h2 className="text-white text-xl mb-2">
                    Segmentation Dataset
                  </h2>
                  <p className="text-slate-400 text-sm">
                    Build datasets with satellite imagery and segmentation masks
                  </p>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-blue-400 rounded-full" />
                    </div>
                    <span>Capture golf course locations</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-blue-400 rounded-full" />
                    </div>
                    <span>Annotate ground truth masks</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-blue-400 rounded-full" />
                    </div>
                    <span>Export for U-Net training</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-blue-400 text-sm group-hover:gap-3 transition-all">
                  <span>Start Building</span>
                  <ArrowRight className="size-4" />
                </div>
              </div>
            </Link>

            {/* Classification Dataset */}
            <Link href="/classification">
              <div className="group relative bg-slate-800 border-2 border-slate-700 rounded-2xl p-6 text-left transition-all hover:border-orange-500 hover:shadow-xl hover:shadow-orange-500/20 hover:-translate-y-1 h-full cursor-pointer">
                <div className="absolute top-4 left-4 px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded-full border border-orange-500/30">
                  Binary Labels
                </div>

                <div className="absolute top-4 right-4 size-10 bg-orange-500/10 rounded-full flex items-center justify-center group-hover:bg-orange-500/20 transition-colors">
                  <Tag className="size-5 text-orange-400" />
                </div>

                <div className="mb-4 mt-8">
                  <h2 className="text-white text-xl mb-2">
                    Classification Dataset
                  </h2>
                  <p className="text-slate-400 text-sm">
                    Label images as Golf or Not Golf for classifier training
                  </p>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-orange-400 rounded-full" />
                    </div>
                    <span>Quick capture and label</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-orange-400 rounded-full" />
                    </div>
                    <span>Golf ✓ or Not Golf ✗</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-orange-400 rounded-full" />
                    </div>
                    <span>Export for classifier training</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-orange-400 text-sm group-hover:gap-3 transition-all">
                  <span>Start Labeling</span>
                  <ArrowRight className="size-4" />
                </div>
              </div>
            </Link>

            {/* Segmentation Analysis */}
            <Link href="/segmentation">
              <div className="group relative bg-slate-800 border-2 border-slate-700 rounded-2xl p-6 text-left transition-all hover:border-emerald-500 hover:shadow-xl hover:shadow-emerald-500/20 hover:-translate-y-1 h-full cursor-pointer">
                <div className="absolute top-4 left-4 px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded-full border border-emerald-500/30">
                  Analysis
                </div>

                <div className="absolute top-4 right-4 size-10 bg-emerald-500/10 rounded-full flex items-center justify-center group-hover:bg-emerald-500/20 transition-colors">
                  <Target className="size-5 text-emerald-400" />
                </div>

                <div className="mb-4 mt-8">
                  <h2 className="text-white text-xl mb-2">
                    Segmentation Analysis
                  </h2>
                  <p className="text-slate-400 text-sm">
                    Analyze golf courses and identify features with AI
                  </p>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-emerald-400 rounded-full" />
                    </div>
                    <span>Search any golf course</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-emerald-400 rounded-full" />
                    </div>
                    <span>AI-powered feature detection</span>
                  </div>
                  <div className="flex items-start gap-2 text-slate-300 text-xs">
                    <div className="size-4 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="size-1.5 bg-emerald-400 rounded-full" />
                    </div>
                    <span>View statistics and overlay</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-emerald-400 text-sm group-hover:gap-3 transition-all">
                  <span>Start Analyzing</span>
                  <ArrowRight className="size-4" />
                </div>
              </div>
            </Link>
          </div>

          {/* About Section */}
          <div className="mt-16 border-t border-slate-800 pt-12">
            <div className="text-center mb-8">
              <h2 className="text-white text-2xl mb-3">About This Project</h2>
              <p className="text-slate-400 max-w-3xl mx-auto">
                A semantic segmentation tool for golf course aerial imagery using deep learning
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {/* Purpose */}
              <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                <h3 className="text-white text-lg mb-3">Purpose</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  This application enables semantic segmentation of golf course satellite imagery,
                  identifying key features such as fairways, greens, bunkers, water hazards, and rough areas.
                  Built using a U-Net architecture with a ResNet50 encoder, the model was trained on
                  high-resolution orthophoto imagery from Danish golf courses.
                </p>
              </div>

              {/* Methodology */}
              <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                <h3 className="text-white text-lg mb-3">Methodology</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  The segmentation model uses MobileNetV2 depthwise separable convolutions for efficient
                  feature extraction. Training data consists of manually annotated golf course orthophotos
                  with 6-class pixel-level labels. The classification component uses UC Merced Land Use
                  dataset for binary golf course detection.
                </p>
              </div>
            </div>

            {/* References */}
            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
              <h3 className="text-white text-lg mb-4">References & Data Sources</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <a
                  href="https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors group"
                >
                  <div className="size-8 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Database className="size-4 text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-200 text-sm font-medium group-hover:text-blue-400 transition-colors flex items-center gap-1">
                      Danish Golf Courses Orthophotos
                      <ExternalLink className="size-3" />
                    </div>
                    <div className="text-slate-500 text-xs">Kaggle Dataset - Training imagery and masks</div>
                  </div>
                </a>

                <a
                  href="https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors group"
                >
                  <div className="size-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Target className="size-4 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-200 text-sm font-medium group-hover:text-purple-400 transition-colors flex items-center gap-1">
                      U-Net Segmentation Notebook
                      <ExternalLink className="size-3" />
                    </div>
                    <div className="text-slate-500 text-xs">Kaggle Notebook - Model architecture reference</div>
                  </div>
                </a>

                <a
                  href="https://huggingface.co/datasets/blanchon/UC_Merced"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors group"
                >
                  <div className="size-8 bg-orange-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Tag className="size-4 text-orange-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-200 text-sm font-medium group-hover:text-orange-400 transition-colors flex items-center gap-1">
                      UC Merced Land Use Dataset
                      <ExternalLink className="size-3" />
                    </div>
                    <div className="text-slate-500 text-xs">HuggingFace - Classification training data</div>
                  </div>
                </a>

                <a
                  href="https://arxiv.org/abs/1704.04861"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors group"
                >
                  <div className="size-8 bg-emerald-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="size-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-200 text-sm font-medium group-hover:text-emerald-400 transition-colors flex items-center gap-1">
                      MobileNets: Efficient CNNs
                      <ExternalLink className="size-3" />
                    </div>
                    <div className="text-slate-500 text-xs">arXiv Paper - Depthwise separable convolutions</div>
                  </div>
                </a>

                <a
                  href="https://ieeexplore.ieee.org/document/10221980"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors group"
                >
                  <div className="size-8 bg-cyan-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="size-4 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-slate-200 text-sm font-medium group-hover:text-cyan-400 transition-colors flex items-center gap-1">
                      IEEE Semantic Segmentation
                      <ExternalLink className="size-3" />
                    </div>
                    <div className="text-slate-500 text-xs">IEEE Paper - Deep learning segmentation methods</div>
                  </div>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Footer />
      <Toaster />
    </div>
  );
}
