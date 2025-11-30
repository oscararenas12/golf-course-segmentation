'use client';

import Link from 'next/link';
import { Database, Target, Tag, ArrowRight } from 'lucide-react';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { Toaster } from '@/components/ui/sonner';

export default function HomePage() {
  return (
    <div className="h-screen flex flex-col bg-slate-950">
      <Header />

      <div className="flex-1 flex items-center justify-center">
        <div className="max-w-6xl w-full px-8">
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
        </div>
      </div>

      <Footer />
      <Toaster />
    </div>
  );
}
