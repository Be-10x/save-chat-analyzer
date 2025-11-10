import React from 'react';

export const HeaderIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-600" width="24" height="24" viewBox="0 0 24 24" strokeWidth="2" stroke="currentColor" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <path d="M3 12h4.5l1.5 -6l4 12l2 -9l1.5 3h4.5"></path>
    </svg>
);

export const UploadIcon: React.FC = () => (
    <svg className="mx-auto h-8 w-8 text-slate-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

const IconWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <div className="flex-shrink-0 bg-indigo-500 text-white rounded-lg p-2">
        {children}
    </div>
);

const SvgIcon: React.FC<{ path: string }> = ({ path }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={path} />
    </svg>
);

export const SummaryIcon: React.FC = () => <IconWrapper><SvgIcon path="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></IconWrapper>;
export const ImprovementIcon: React.FC = () => <IconWrapper><SvgIcon path="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></IconWrapper>;
export const ActionIcon: React.FC = () => <IconWrapper><SvgIcon path="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></IconWrapper>;
export const ProblemIcon: React.FC = () => <IconWrapper><SvgIcon path="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></IconWrapper>;
export const HighlightIcon: React.FC = () => <IconWrapper><SvgIcon path="M5 3v4M3 5h4M4 17v4M2 19h4M17 3v4M15 5h4M17 17v4M15 19h4M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></IconWrapper>;
export const LogIcon: React.FC = () => <IconWrapper><SvgIcon path="M4 6h16M4 12h16M4 18h7" /></IconWrapper>;
export const UserIcon: React.FC = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" /></svg>;
export const QuoteIcon: React.FC = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" /></svg>;
