'use client';

import NeuralNetworkVisualization from "@/components/NeuralNetworkVisualization";

export default function NeuralNetworkDemo() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="p-6">
        <h1 className="text-3xl font-bold mb-2">Neural Network Visualization</h1>
        <p className="text-gray-700 mb-6">
          Interactive visualization of a neural network with randomized weights and activations.
          Yellow neurons represent the input layer, while magenta neurons represent the output layer.
        </p>
      </div>
      
      <div className="flex-1 w-full">
        <NeuralNetworkVisualization />
      </div>
    </div>
  );
}