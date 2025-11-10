
import React, { useState, useRef } from 'react';
import type { AnalysisReportData } from './types';
import { analyzeChatLog } from './services/geminiService';
import ChatInputForm from './components/ChatInputForm';
import AnalysisReport from './components/AnalysisReport';
import { HeaderIcon } from './components/icons';

interface QuotaErrorDetails {
  metric: string;
  limit: string;
  retryDelay: string;
}

// --- New Progress Indicator Component ---
interface ProgressIndicatorProps {
  progress: number;
  message: string;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ progress, message }) => {
  const roundedProgress = Math.round(progress);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (roundedProgress / 100) * circumference;

  return (
    <div 
        className="flex flex-col items-center justify-center bg-slate-800 p-8 rounded-lg shadow-xl" 
        aria-live="polite"
        style={{backgroundColor: '#1e293b'}}
    >
      
      <div className="relative h-28 w-28">
        <svg className="h-full w-full" viewBox="0 0 100 100">
          {/* Background track */}
          <circle
            className="stroke-current text-slate-700"
            strokeWidth="8"
            cx="50"
            cy="50"
            r={radius}
            fill="transparent"
          />
          {/* Progress ring */}
          <circle
            className="stroke-current text-indigo-500 transition-all duration-500 ease-in-out"
            strokeWidth="8"
            strokeLinecap="round"
            cx="50"
            cy="50"
            r={radius}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
          />
          {/* Percentage Text */}
          <text
            x="50"
            y="52"
            fontFamily="sans-serif"
            fontSize="24"
            fontWeight="bold"
            textAnchor="middle"
            alignmentBaseline="middle"
            className="fill-current text-slate-200"
          >
            {`${roundedProgress}`}<tspan fontSize="12" dy="-0.6em">%</tspan>
          </text>
        </svg>
      </div>

      <h2 className="mt-6 text-xl font-bold text-white">
        Analyzing chat log...
      </h2>
      <p className="mt-2 text-center text-sm text-slate-400 max-w-xs">
         This may take a moment. We're extracting key insights for you.
      </p>
       <p className="mt-4 h-4 text-sm font-medium text-slate-300">
        {message}
      </p>
    </div>
  );
};
// --- End of Component ---


const App: React.FC = () => {
  const [report, setReport] = useState<AnalysisReportData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [quotaErrorDetails, setQuotaErrorDetails] = useState<QuotaErrorDetails | null>(null);

  // State for progress simulation
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const progressIntervalRef = useRef<number | null>(null);

  const stopProgress = () => {
    if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
    }
  };

  const startProgress = () => {
      stopProgress(); // Ensure no multiple intervals running
      setProgress(0);
      setProgressMessage("Initiating analysis...");

      progressIntervalRef.current = window.setInterval(() => {
          setProgress(prev => {
              let newProgress = prev;
              if (prev < 95) {
                  // Simulate fast start, then slow down
                  const increment = Math.random() * (15 - prev / 10);
                  newProgress = Math.min(prev + increment, 95);
              } else {
                  // Creep towards 99 at the end
                  newProgress = Math.min(prev + 0.5, 99);
              }

              if (newProgress >= 0 && newProgress < 20) {
                  setProgressMessage("Parsing chat log...");
              } else if (newProgress >= 20 && newProgress < 45) {
                  setProgressMessage("Filtering conversational noise...");
              } else if (newProgress >= 45 && newProgress < 75) {
                  setProgressMessage("Identifying key themes & questions...");
              } else if (newProgress >= 75 && newProgress < 95) {
                  setProgressMessage("Synthesizing insights...");
              } else {
                  setProgressMessage("Finalizing report...");
              }
              
              return newProgress;
          });
      }, 500);
  };


  const handleAnalyze = async (chatLog: string, instructorNames: string) => {
    if (!chatLog.trim()) {
      setError('Chat log cannot be empty.');
      return;
    }
    if (!instructorNames.trim()) {
      setError('Please specify at least one instructor/host name.');
      return;
    }

    setIsLoading(true);
    startProgress(); // Start the progress simulation
    setError(null);
    setReport(null);
    setQuotaErrorDetails(null);

    try {
      const result = await analyzeChatLog(chatLog, instructorNames);
      setReport(result);
    } catch (e) {
      console.error(e);
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred during analysis.';
      
      let isQuotaError = false;
      
      try {
        if (errorMessage.includes('RESOURCE_EXHAUSTED') || errorMessage.includes('429')) {
          const errorJson = JSON.parse(errorMessage);
          const errorDetails = errorJson.error || {};
          if (errorDetails.status === 'RESOURCE_EXHAUSTED' || errorDetails.code === 429) {
            isQuotaError = true;
            const details = errorDetails.details || [];
            const quotaFailure = details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.QuotaFailure');
            const retryInfo = details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
            
            const violation = quotaFailure?.violations?.[0];
            
            setQuotaErrorDetails({
              metric: violation?.quotaMetric?.replace('generate_content_', '').replace('_token_count', ' tokens') || 'Unknown metric',
              limit: violation?.quotaValue ? `${parseInt(violation.quotaValue).toLocaleString()}` : 'Not specified',
              retryDelay: retryInfo?.retryDelay?.replace('s', ' seconds') || 'a moment'
            });
          }
        }
      } catch (parseError) {
        // Not a parsable JSON error, fall through to generic handling.
      }

      if (!isQuotaError) {
         if (errorMessage.includes('API key not valid') ||
            errorMessage.includes('invalid') ||
            errorMessage.includes('API Key must be set') ||
            errorMessage.includes('Requested entity was not found')) {
          setError("API key configuration error."); // Simplified message trigger
        } else {
          setError(errorMessage);
        }
      }

    } finally {
      stopProgress();
      setProgress(100); // Jump to 100 on completion
      setTimeout(() => {
          setIsLoading(false);
      }, 300); // Small delay to show 100%
    }
  };
  
  const handleReset = () => {
    setReport(null);
    setError(null);
    setIsLoading(false);
    setQuotaErrorDetails(null);
    stopProgress();
    setProgress(0);
  }
  
  const renderContent = () => {
    return (
      <>
        {isLoading && (
          <ProgressIndicator progress={progress} message={progressMessage} />
        )}
        
        {report && !isLoading && (
          <div>
            <AnalysisReport report={report} />
            <div className="mt-8 text-center">
               <button 
                onClick={handleReset}
                className="bg-indigo-600 text-white font-semibold py-2 px-6 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
                  Analyze Another Chat
               </button>
            </div>
          </div>
        )}

        {!report && !isLoading && (
          <ChatInputForm onAnalyze={handleAnalyze} disabled={isLoading} />
        )}
      </>
    );
  }

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-sans">
      <header className="bg-white dark:bg-slate-800/50 shadow-sm sticky top-0 z-10 backdrop-blur-md">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <HeaderIcon />
              <h1 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
                Save Chat Analyzer
              </h1>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4 sm:p-6 lg:p-8">
        <div className="max-w-4xl mx-auto">
          {quotaErrorDetails && (
            <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 text-amber-800 dark:text-amber-200 p-4 sm:p-6 mb-6 rounded-lg shadow" role="alert">
              <p className="font-bold text-lg">API Quota Exceeded</p>
              <div className="mt-2 text-slate-700 dark:text-slate-300">
                <p>The analysis failed because the chat log is too large for the current API plan (Free Tier).</p>
                <p className="mt-4 font-semibold">What you can do:</p>
                <ul className="list-disc list-inside mt-2 space-y-2 text-sm">
                    <li>
                        <strong>Reduce Input Size:</strong> Try again with a smaller section of the chat log.
                    </li>
                    <li>
                        <strong>Upgrade Your Plan:</strong> For large-scale analysis, you can upgrade your Gemini API plan to increase the rate limits.
                        <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline ml-1">
                            Manage your billing details.
                        </a>
                    </li>
                </ul>
                
                <div className="mt-4 pt-3 border-t border-amber-300 dark:border-amber-700/50">
                  <p className="text-xs font-semibold text-slate-600 dark:text-slate-400">TECHNICAL DETAILS:</p>
                  <ul className="text-xs font-mono bg-amber-100 dark:bg-amber-900/30 p-2 rounded mt-1 text-slate-600 dark:text-slate-300 space-y-1">
                    <li><span className="font-semibold">Limit Type:</span> {quotaErrorDetails.metric}</li>
                    <li><span className="font-semibold">Current Limit:</span> {quotaErrorDetails.limit} per minute</li>
                    <li><span className="font-semibold">Suggested Retry:</span> Wait {quotaErrorDetails.retryDelay} and try again.</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {error && !error.includes("API key") && (
             <div className="bg-red-100 dark:bg-red-900/20 border-l-4 border-red-500 text-red-800 dark:text-red-200 p-4 sm:p-6 mb-6 rounded-lg shadow" role="alert">
              <p className="font-bold text-lg">Analysis Failed</p>
              <div className="mt-2 text-slate-700 dark:text-slate-300">
                <p>We couldn't generate the report from the input, either because:</p>
                <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
                    <li>The data you submitted (chat log or file) was not in the correct format, was too long, or contained content the AI could not process.</li>
                    <li>There is a temporary issue or outage with the AI backend (Gemini API).</li>
                </ul>
                <p className="mt-4">Please check your input and try again. If the problem persists, the service may be temporarily unavailable.</p>
                
                <div className="mt-4 pt-3 border-t border-red-300 dark:border-red-700/50">
                  <p className="text-xs font-semibold text-slate-600 dark:text-slate-400">TECHNICAL DETAILS:</p>
                  <p className="text-xs font-mono bg-red-50 dark:bg-red-900/30 p-2 rounded mt-1 text-slate-600 dark:text-slate-300">
                    {error}
                  </p>
                </div>
              </div>
            </div>
          )}
          {error && error.includes("API key") && (
             <div className="bg-red-100 dark:bg-red-900/20 border-l-4 border-red-500 text-red-800 dark:text-red-200 p-4 sm:p-6 mb-6 rounded-lg shadow" role="alert">
              <p className="font-bold text-lg">API Key Configuration Error</p>
              <div className="mt-2 text-slate-700 dark:text-slate-300">
                <p>The application could not connect to the Gemini API. This is usually due to a missing or incorrect API key in the deployment settings.</p>
                <p className="mt-4 font-semibold">How to fix this:</p>
                <ul className="list-disc list-inside mt-2 space-y-2 text-sm">
                    <li>
                        Go to your project's settings on Vercel and find the <strong>Environment Variables</strong> section.
                        <a href="https://vercel.com/dashboard" target="_blank" rel="noopener noreferrer" className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline ml-1">
                            Open Vercel Dashboard
                        </a>
                    </li>
                    <li>
                        Ensure there is a variable named exactly <code className="text-indigo-600 dark:text-indigo-400 bg-slate-100 dark:bg-slate-700/50 px-1 py-0.5 rounded">VITE_API_KEY</code>.
                    </li>
                    <li>
                        Paste your new "central billing API key" from your developer into the 'Value' field for this variable.
                    </li>
                    <li>
                        <strong>Crucially, you must redeploy the application</strong> after saving the variable for the change to take effect. You can do this from the "Deployments" tab in Vercel.
                    </li>
                </ul>
              </div>
            </div>
          )}
          
          {renderContent()}
          
        </div>
      </main>
      
      <footer className="text-center py-4 mt-8">
        <p className="text-sm text-slate-500 dark:text-slate-400">Powered by Gemini API</p>
      </footer>
    </div>
  );
};

export default App;
