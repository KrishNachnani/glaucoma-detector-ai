'use client';

import React, { useEffect, useRef } from 'react';

interface NeuronData {
  x: number;
  y: number;
  radius: number;
  color: string;
  originalColor: string;
  value: number;
  hovered: boolean;
  active: boolean;
}

interface ConnectionData {
  from: { layer: number; index: number };
  to: { layer: number; index: number };
  weight: number;
  signal: number;
  active: boolean;
}

interface LayerData {
  size: number;
  color: string;
  name: string;
}

const NeuralNetworkVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animatingRef = useRef(false);
  const animationFrameRef = useRef(0);
  const animationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const neuronsRef = useRef<NeuronData[][]>([]);
  const connectionsRef = useRef<ConnectionData[][]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Neural network parameters
    const layers: LayerData[] = [
      { size: 10, color: '#FFDD00', name: 'Input Layer' },   // Input layer (yellow)
      { size: 8, color: '#cfc8c8', name: 'Hidden Layer 1' }, // Hidden layer 1 (gray)
      { size: 10, color: '#cfc8c8', name: 'Hidden Layer 2' }, // Hidden layer 2 (gray)
      { size: 6, color: '#cfc8c8', name: 'Hidden Layer 3' }, // Hidden layer 3 (gray)
      { size: 2, color: '#FF00FF', name: 'Output Layer' }    // Output layer (magenta)
    ];

    // Store the original colors
    const originalColors = layers.map(layer => layer.color);
    
    // Spacing configuration
    const spacing = 80;
    const layerSpacing = 200;
    const animationDuration = 150;

    // Set canvas size
    const resizeCanvas = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        initializeNeurons();
        initializeConnections();
        drawNetwork();
      }
    };

    // Initialize neurons with positions
    const initializeNeurons = () => {
      const neurons: NeuronData[][] = [];
      
      const totalNetworkWidth = (layers.length - 1) * layerSpacing;
      const startX = (canvas.width - totalNetworkWidth) / 2;
      
      layers.forEach((layer, layerIndex) => {
        const layerHeight = (layer.size - 1) * spacing;
        const startY = (canvas.height - layerHeight) / 2;
        
        const layerNeurons: NeuronData[] = [];
        
        for (let i = 0; i < layer.size; i++) {
          layerNeurons.push({
            x: startX + layerIndex * layerSpacing,
            y: startY + i * spacing,
            radius: 15,
            color: layer.color,
            originalColor: layer.color,
            value: 0, // Neuron activation value
            hovered: false,
            active: false
          });
        }
        
        neurons.push(layerNeurons);
      });
      
      neuronsRef.current = neurons;
    };
    
    // Initialize connections with random weights
    const initializeConnections = () => {
      const connections: ConnectionData[][] = [];
      const neurons = neuronsRef.current;
      
      for (let i = 0; i < neurons.length - 1; i++) {
        const layerConnections: ConnectionData[] = [];
        
        for (let j = 0; j < neurons[i].length; j++) {
          for (let k = 0; k < neurons[i + 1].length; k++) {
            layerConnections.push({
              from: { layer: i, index: j },
              to: { layer: i + 1, index: k },
              weight: Math.random() * 2 - 1, // Weight between -1 and 1
              signal: 0, // Signal passing through connection
              active: false // Whether connection is highlighted
            });
          }
        }
        
        connections.push(layerConnections);
      }
      
      connectionsRef.current = connections;
    };
    
    // Draw a connection
    const drawConnection = (conn: ConnectionData) => {
      if (!ctx) return;
      
      const neurons = neuronsRef.current;
      const fromNeuron = neurons[conn.from.layer][conn.from.index];
      const toNeuron = neurons[conn.to.layer][conn.to.index];
      
      ctx.beginPath();
      ctx.moveTo(fromNeuron.x, fromNeuron.y);
      ctx.lineTo(toNeuron.x, toNeuron.y);
      
      if (conn.active) {
        const signalStrength = Math.abs(conn.signal);
        const width = 1 + 5 * signalStrength;
        
        if (conn.signal > 0) {
          ctx.strokeStyle = `rgba(0, 200, 0, ${0.3 + 0.7 * signalStrength})`;
        } else {
          ctx.strokeStyle = `rgba(255, 0, 0, ${0.3 + 0.7 * signalStrength})`;
        }
        
        ctx.lineWidth = width;
      } else {
        const weight = Math.abs(conn.weight);
        const width = 0.5 + 2 * weight;
        
        if (conn.weight > 0) {
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
        } else {
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.2)';
        }
        
        ctx.lineWidth = width;
      }
      
      ctx.stroke();
    };
    
    // Draw a neuron
    const drawNeuron = (neuron: NeuronData) => {
      if (!ctx) return;
      
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
      
      // Determine which color to use
      const useColor = neuron.active ? '#0088ff' : neuron.color;
      
      // Fill based on activation
      const activation = Math.max(0, Math.min(1, (neuron.value + 1) / 2)); // Map from [-1,1] to [0,1]
      const r = parseInt(useColor.slice(1, 3), 16);
      const g = parseInt(useColor.slice(3, 5), 16);
      const b = parseInt(useColor.slice(5, 7), 16);
      
      if (neuron.hovered) {
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
        ctx.shadowColor = useColor;
        ctx.shadowBlur = 15;
      } else {
        const brightness = 0.3 + 0.7 * activation;
        ctx.fillStyle = `rgba(${r * brightness}, ${g * brightness}, ${b * brightness}, 0.9)`;
        ctx.shadowBlur = 0;
      }
      
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Draw outline
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#333';
      ctx.stroke();
      
      // Draw activation value
      if (neuron.hovered) {
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(neuron.value.toFixed(2), neuron.x, neuron.y);
      }
    };
    
    // Draw the entire network
    const drawNetwork = () => {
      if (!ctx || !canvas) return;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections
      connectionsRef.current.forEach(layerConns => {
        layerConns.forEach(conn => {
          drawConnection(conn);
        });
      });
      
      // Draw neurons
      neuronsRef.current.forEach(layer => {
        layer.forEach(neuron => {
          drawNeuron(neuron);
        });
      });
      
      // Draw layer names
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#ffffff';
      
      layers.forEach((layer, i) => {
        if (neuronsRef.current[i] && neuronsRef.current[i].length > 0) {
          const x = neuronsRef.current[i][0].x;
          const y = neuronsRef.current[i][0].y - 50;
          ctx.fillText(layer.name, x, y);
        }
      });
    };
    
    // Animate a forward pass through the network
    const animateForwardPass = () => {
      if (!animatingRef.current) return;
      
      // Reset all signals
      connectionsRef.current.forEach(layerConns => {
        layerConns.forEach(conn => {
          conn.active = false;
          conn.signal = 0;
        });
      });
      
      // Reset all neuron active states at the beginning of a cycle
      if (animationFrameRef.current === 0) {
        neuronsRef.current.forEach(layer => {
          layer.forEach(neuron => {
            neuron.active = false;
          });
        });
        
        // Generate random inputs
        neuronsRef.current[0].forEach(neuron => {
          neuron.value = Math.random() * 2 - 1; // Between -1 and 1
        });
      }
      
      // Propagate through network
      const currentLayer = Math.floor(animationFrameRef.current / 30);
      
      if (currentLayer < connectionsRef.current.length) {
        const layerProgress = (animationFrameRef.current % 30) / 30;
        
        // Deactivate previous layer if we're moving to a new layer
        if (layerProgress < 0.1 && currentLayer > 0) {
          neuronsRef.current[currentLayer - 1].forEach(neuron => {
            neuron.active = false;
          });
        }
        
        // Activate neurons in the current layer
        neuronsRef.current[currentLayer].forEach(neuron => {
          neuron.active = true;
        });
        
        // Activate neurons in the next layer as signals reach them
        if (layerProgress > 0.7 && currentLayer < neuronsRef.current.length - 1) {
          // Special handling for the output layer (last layer)
          if (currentLayer === connectionsRef.current.length - 1) {
            // Calculate values for output neurons before deciding which one to activate
            const outputLayer = neuronsRef.current[neuronsRef.current.length - 1];
            
            // If we're at the transition to output layer, find the neuron with highest value
            if (layerProgress > 0.9) {
              let maxValue = -Infinity;
              let maxIndex = 0;
              
              // Find the neuron with the highest activation value
              outputLayer.forEach((neuron, index) => {
                if (neuron.value > maxValue) {
                  maxValue = neuron.value;
                  maxIndex = index;
                }
                // Set all neurons to inactive initially
                neuron.active = false;
              });
              
              // Activate only the neuron with the highest value
              if (maxValue > -Infinity) {
                outputLayer[maxIndex].active = true;
              }
            } else {
              // During transition, temporarily activate all neurons
              outputLayer.forEach(neuron => {
                neuron.active = true;
              });
            }
          } else {
            // For non-output layers, activate all neurons normally
            neuronsRef.current[currentLayer + 1].forEach(neuron => {
              neuron.active = true;
            });
          }
          
          // Deactivate current layer as we transition to the next
          if (layerProgress > 0.9) {
            neuronsRef.current[currentLayer].forEach(neuron => {
              neuron.active = false;
            });
          }
        }
        
        connectionsRef.current[currentLayer].forEach(conn => {
          // Randomly skip some paths based on weight - stronger weights have higher chance of activating
          const activationThreshold = 0.3; // Minimum weight to have a chance of activating
          const weightMagnitude = Math.abs(conn.weight);
          
          // Skip this connection if the weight is too low or random chance based on weight
          if (weightMagnitude < activationThreshold || Math.random() > weightMagnitude * 1.5) {
            return;
          }
          
          const fromNeuron = neuronsRef.current[conn.from.layer][conn.from.index];
          conn.active = true;
          conn.signal = fromNeuron.value * conn.weight * layerProgress;
          
          // Update target neuron gradually
          if (layerProgress > 0.5) {
            const toNeuron = neuronsRef.current[conn.to.layer][conn.to.index];
            toNeuron.value = Math.tanh(conn.signal); // Apply activation function
          }
        });
      }
      
      drawNetwork();
      
      animationFrameRef.current++;
      
      if (animationFrameRef.current < animationDuration) {
        requestAnimationFrame(animateForwardPass);
      } else {
        // For the last frame, ensure only one output neuron is active
        if (neuronsRef.current.length > 0) {
          const lastLayer = neuronsRef.current.length - 1;
          const outputLayer = neuronsRef.current[lastLayer];
          
          // Find the neuron with the highest value
          let maxValue = -Infinity;
          let maxIndex = 0;
          
          outputLayer.forEach((neuron, index) => {
            if (neuron.value > maxValue) {
              maxValue = neuron.value;
              maxIndex = index;
            }
            // Deactivate all neurons initially
            neuron.active = false;
          });
          
          // Activate only the winning neuron
          if (outputLayer.length > 0) {
            outputLayer[maxIndex].active = true;
          }
        }
        
        drawNetwork(); // Redraw to show deactivated neurons
        
        animationFrameRef.current = 0;
        if (animatingRef.current) {
          // Continue with a 0.5 second delay
          if (animationTimeoutRef.current) {
            clearTimeout(animationTimeoutRef.current);
          }
          animationTimeoutRef.current = setTimeout(() => {
            if (animatingRef.current) {
              requestAnimationFrame(animateForwardPass);
            }
          }, 500); // 500ms delay
        }
      }
    };

    // Mouse interaction handler
    const handleMouseMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
      
      // Check if mouse is over any neuron
      let hoveredFound = false;
      
      neuronsRef.current.forEach(layer => {
        layer.forEach(neuron => {
          const dx = neuron.x - mouseRef.current.x;
          const dy = neuron.y - mouseRef.current.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < neuron.radius) {
            neuron.hovered = true;
            hoveredFound = true;
          } else {
            neuron.hovered = false;
          }
        });
      });
      
      if (hoveredFound) {
        canvas.style.cursor = 'pointer';
      } else {
        canvas.style.cursor = 'default';
      }
      
      drawNetwork();
    };

    // Start animation function
    const startAnimation = () => {
      if (!animatingRef.current) {
        animatingRef.current = true;
        animationFrameRef.current = 0;
        animateForwardPass();
      }
    };

    // Stop animation function
    const stopAnimation = () => {
      animatingRef.current = false;
      if (animationTimeoutRef.current) {
        clearTimeout(animationTimeoutRef.current);
        animationTimeoutRef.current = null;
      }
    };

    // Reset function
    const resetNetwork = () => {
      stopAnimation();
      initializeNeurons();
      initializeConnections();
      
      // Reset all neuron values
      neuronsRef.current.forEach(layer => {
        layer.forEach(neuron => {
          neuron.value = 0;
          neuron.active = false;
        });
      });
      
      drawNetwork();
    };

    // Event listeners
    window.addEventListener('resize', resizeCanvas);
    canvas.addEventListener('mousemove', handleMouseMove);
    
    // Initialize the network
    resizeCanvas();
    initializeNeurons();
    initializeConnections();
    drawNetwork();

    // Auto-start animation
    startAnimation();
    
    // Cleanup function
    return () => {
      stopAnimation();
      window.removeEventListener('resize', resizeCanvas);
      if (canvas) {
        canvas.removeEventListener('mousemove', handleMouseMove);
      }
    };
  }, []);

  return (
    <div className="neural-network-container relative w-full h-full min-h-[500px]">
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
{/*       
      <div className="controls absolute top-5 right-5 bg-white/80 p-2.5 rounded-lg shadow-md">
        <button 
          onClick={() => {
            animatingRef.current = true;
            animationFrameRef.current = 0;
            requestAnimationFrame(() => {
              if (animatingRef.current) {
                const animateForwardPass = () => {
                  // The animation is already defined in the useEffect
                };
                animateForwardPass();
              }
            });
          }}
          className="bg-blue-600 text-white border-none py-2 px-3 rounded cursor-pointer mx-1 font-bold hover:bg-blue-700"
        >
          Start Animation
        </button>
        <button 
          onClick={() => {
            animatingRef.current = false;
            if (animationTimeoutRef.current) {
              clearTimeout(animationTimeoutRef.current);
              animationTimeoutRef.current = null;
            }
          }}
          className="bg-blue-600 text-white border-none py-2 px-3 rounded cursor-pointer mx-1 font-bold hover:bg-blue-700"
        >
          Stop Animation
        </button>
        <button 
          onClick={() => {
            animatingRef.current = false;
            if (animationTimeoutRef.current) {
              clearTimeout(animationTimeoutRef.current);
              animationTimeoutRef.current = null;
            }
            
            // Reset and initialize again
            const canvas = canvasRef.current;
            if (!canvas) return;
            
            const resizeCanvas = () => {
              // This function is already defined in the useEffect
            };
            
            resizeCanvas();
          }}
          className="bg-blue-600 text-white border-none py-2 px-3 rounded cursor-pointer mx-1 font-bold hover:bg-blue-700"
        >
          Reset
        </button>
      </div> */}
      
      {/* <div className="legend absolute bottom-5 right-5 bg-white/80 p-2.5 rounded-lg shadow-md">
        <h3 className="mt-0">Legend</h3>
        <div className="flex items-center my-1.5">
          <div className="w-5 h-5 mr-2.5 rounded bg-[#FFDD00]"></div>
          <span>Input Layer (10 neurons)</span>
        </div>
        <div className="flex items-center my-1.5">
          <div className="w-5 h-5 mr-2.5 rounded bg-[#cfc8c8]"></div>
          <span>Hidden Layers (inactive)</span>
        </div>
        <div className="flex items-center my-1.5">
          <div className="w-5 h-5 mr-2.5 rounded bg-[#0088ff]"></div>
          <span>Currently Active Neurons</span>
        </div>
        <div className="flex items-center my-1.5">
          <div className="w-5 h-5 mr-2.5 rounded bg-[#FF00FF]"></div>
          <span>Output Layer (2 neurons)</span>
        </div>
      </div> */}
    </div>
  );
};

export default NeuralNetworkVisualization;