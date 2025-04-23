'use client';

import { useState } from 'react';
import { Upload, Eye, AlertCircle } from 'lucide-react';
import Image from 'next/image';
import NeuralNetworkVisualization from "@/components/NeuralNetworkVisualization";

interface AnalysisResult {
  filename: string;
  internal_filename: string;
  features: {
    feature_0: number;
    feature_1: number;
    feature_2: number;
  };
  prediction: string;
  probability: {
    "Glaucoma": number;
    "No Glaucoma": number;
  },
  visualization_image: string;
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Upload to API
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use environment variable for API URL
      //const apiUrl = process.env.NEXT_PUBLIC_GLAUCOMA_API_URL; 
      const apiUrl="http://localhost:8999";
      console.log('API URL:', apiUrl);
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240]">
      <div className="max-w-7xl mx-auto px-4 py-12 grid md:grid-cols-2 gap-12 items-center">
        <div className="space-y-6">
          <h1 className="text-5xl font-bold text-white leading-tight">
            Glaucoma Detection
            <span className="block text-blue-400">Using AI Technology</span>
          </h1>
          <p className="text-gray-300 text-lg">
            Upload your retinal image for instant AI-powered glaucoma detection. Our advanced
            system provides quick and accurate analysis to support early diagnosis.
          </p>
          
          <div className="relative">
            <input
              type="file"
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="flex items-center justify-center w-full h-64 border-2 border-dashed border-blue-400 rounded-xl cursor-pointer hover:border-blue-500 transition-all bg-[#1a2942]/50"
            >
              {selectedImage ? (
                <div className={`relative w-full h-full ${result?.prediction === 'Glaucoma' ? 'ring-4 ring-red-500' : result ? 'ring-4 ring-green-500' : ''}`}>
                  <Image
                    src={selectedImage}
                    alt="Uploaded image"
                    fill
                    className="object-contain p-2 rounded-xl"
                  />
                </div>
              ) : (
                <div className="text-center space-y-4">
                  <Upload className="w-12 h-12 text-blue-400 mx-auto" />
                  <p className="text-blue-400">Click or drag image to upload</p>
                </div>
              )}
            </label>
          </div>
        </div>
        
        {/* Show the glaucoma image when no image is selected, otherwise show analysis results */}
        {!selectedImage ? (
          <div className="bg-[#1a2942] p-8 rounded-2xl space-y-6">
            <h2 className="text-2xl font-bold text-white">Glaucoma Information</h2>
            <div className="relative w-full h-64">
              <Image 
                src="/images/glaucoma.jpg"
                alt="Glaucoma Illustration"
                fill
                className="object-contain rounded-xl"
              />
            </div>
            <p className="text-gray-300">
              Glaucoma is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye. 
              Early detection is crucial to prevent vision loss.
            </p>
          </div>
        ) : (
          <div className="bg-[#1a2942] p-8 rounded-2xl space-y-6">
            <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
            
            {isAnalyzing ? (
              <div className="flex-2 h-full w-full"> 
                <br></br><br></br><br></br><br></br>
                <NeuralNetworkVisualization />
              </div>
            ) : result ? (
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${
                  result.prediction === 'Glaucoma' 
                    ? 'bg-red-500/20 border border-red-500' 
                    : 'bg-green-500/20 border border-green-500'
                }`}>
                  <div className="flex items-center gap-2">
                    <AlertCircle className={result.prediction === 'Glaucoma' ? 'text-red-500' : 'text-green-500'} />
                    <span className="text-xl font-semibold text-white">
                      {result.prediction === 'Glaucoma' ? 'Glaucoma Detected' : 'No Glaucoma Detected'}
                    </span>
                  </div>
                  <p className="text-gray-300 mt-2">
                    Confidence: {result.prediction === 'Glaucoma' 
                      ? Math.round(result.probability["Glaucoma"] * 100) 
                      : Math.round(result.probability["No Glaucoma"] * 100)}%
                  </p>
                </div>
                
                <br></br>
                {result.visualization_image && (
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-3">Visualization</h3>
                    <div className="relative h-48 w-full overflow-hidden rounded-lg">
                      <Image 
                        src={`data:image/png;base64,${result.visualization_image}`}
                        alt="Visualization of analysis"
                        fill
                        className="object-contain"
                      />
                    </div>
                    <div className="space-y-4">
                      <section className="bg-[#1a2942] p-8 rounded-xl">
                        <h2 className="text-xl font-semibold text-white mb-4">Grad-CAM Visualizations</h2>
                        <p className="text-gray-300">
                          Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to visualize which parts of an image are most important for a Convolutional Neural Network's (CNN's) decision in an a glaucoma classification task.
                          Heatmaps highlights parts of the eye image that the model is focusing on when making a glaucoma classification. 
                        </p>
                        <br></br>
                        <p className="text-gray-300">
                          Three panels are <b>original image</b>, <b>Grad-CAM heatmap</b>, and a <b>superimposed</b> view.
                        </p>
                      </section>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <Eye className="w-12 h-12 text-blue-400 mb-4" />
                <p className="text-gray-300">
                  Processing your image. Analysis results will appear here shortly.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}