
import React, { useState, useCallback, useRef } from 'react';
import { UploadIcon } from './icons';

interface ChatInputFormProps {
  onAnalyze: (chatLog: string, instructorNames: string) => void;
  disabled: boolean;
}

const SAMPLE_LOG = `08:00:15 Instructor: Good morning everyone! Let's get started.
08:00:45 Priya S.: hi
08:01:10 Rahul M.: hello
08:02:30 Instructor: Today we'll cover Module 3. You can find the files in the usual folder.
08:03:00 Priya S.: where is the folder? i can't find it
08:03:15 Arjun V.: same here, the link seems broken
08:04:00 Instructor: Apologies, let me post the correct link. Here it is: [new link]
08:04:30 Kirti P.: my audio is lagging a bit
08:05:00 Arjun V.: got it, thanks!
08:10:20 Instructor: Okay, now look at line 45, the 'const result = ...'
08:11:00 Rahul M.: I'm getting an 'undefined variable' error on that line
08:11:15 Sonia B.: me too, what did i miss?
08:15:45 Instructor: Ah, make sure you defined the variable on line 32.
08:20:10 Aman G.: wow that shortcut trick is amazing!
08:20:30 Riya T.: +1 really useful
08:25:50 Sonia B.: This real-world example is very clear, thank you!
08:30:00 Instructor: Glad you're finding it helpful! Any questions?
08:31:00 Priya S.: can you please go a bit slower? this part is confusing
08:32:15 Kirti P.: yeah the pace is a little fast for me too`;

const ChatInputForm: React.FC<ChatInputFormProps> = ({ onAnalyze, disabled }) => {
  const [chatLog, setChatLog] = useState<string>('');
  const [instructorNames, setInstructorNames] = useState<string>('Team Be10x, Aditya Goenka, Aditya Kachave');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result;
        setChatLog(typeof text === 'string' ? text : '');
      };
      reader.readAsText(file);
    }
  };
  
  const handleDragOver = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);

  const handleDrop = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
     if (file && file.type === "text/plain") {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result;
        setChatLog(typeof text === 'string' ? text : '');
      };
      reader.readAsText(file);
    }
  }, []);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onAnalyze(chatLog, instructorNames);
  };

  const buttonText = disabled ? 'Analyzing...' : 'Analyze Chat';

  return (
    <div className="bg-white dark:bg-slate-800 p-6 sm:p-8 rounded-xl shadow-lg">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="instructors" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Instructor / Host Names (comma-separated)
          </label>
          <input
            id="instructors"
            type="text"
            value={instructorNames}
            onChange={(e) => setInstructorNames(e.target.value)}
            className="block w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md shadow-sm placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            placeholder="e.g., Team Be10x, Aditya Goenka"
            required
            aria-describedby="instructors-description"
          />
          <div id="instructors-description" className="mt-3 text-xs text-slate-500 dark:text-slate-400 space-y-2">
            <p>
                <strong>Why is this required?</strong> The AI will completely ignore any messages sent by the names you list here. This is crucial for filtering out instructor monologue and focusing the analysis exclusively on learner feedback.
            </p>
            <p>
                <strong>Example:</strong> <code className="text-indigo-600 dark:text-indigo-400 bg-slate-100 dark:bg-slate-700/50 px-1 py-0.5 rounded">Team Be10x, Aditya Goenka, Aditya Kachave</code>
            </p>
          </div>
        </div>

        <div>
           <label htmlFor="chat-log" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Paste Chat Log or Upload File
          </label>
          <textarea
            id="chat-log"
            rows={12}
            value={chatLog}
            onChange={(e) => setChatLog(e.target.value)}
            className="block w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-md shadow-sm placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            placeholder={SAMPLE_LOG}
          />
        </div>
        
        <div className="relative">
            <div className="absolute inset-0 flex items-center" aria-hidden="true">
                <div className="w-full border-t border-slate-300 dark:border-slate-600"></div>
            </div>
            <div className="relative flex justify-center">
                <span className="bg-white dark:bg-slate-800 px-2 text-sm text-slate-500 dark:text-slate-400">Or</span>
            </div>
        </div>

        <label 
          htmlFor="file-upload" 
          className="relative flex justify-center w-full h-24 px-6 pt-5 pb-6 border-2 border-slate-300 dark:border-slate-600 border-dashed rounded-md cursor-pointer hover:border-indigo-500 dark:hover:border-indigo-400 transition-colors"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <div className="space-y-1 text-center">
            <UploadIcon />
            <div className="flex text-sm text-slate-600 dark:text-slate-400">
              <span className="relative font-medium text-indigo-600 dark:text-indigo-400 hover:text-indigo-500">
                Upload a .txt file
              </span>
              <input id="file-upload" name="file-upload" type="file" className="sr-only" accept=".txt" ref={fileInputRef} onChange={handleFileChange} />
              <p className="pl-1">or drag and drop</p>
            </div>
          </div>
        </label>

        <div>
          <button
            type="submit"
            disabled={disabled}
            className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-slate-400 dark:disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors"
          >
            {buttonText}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInputForm;