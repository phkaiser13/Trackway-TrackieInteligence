
import React from 'react';

// Generic Icon wrapper props
interface IconProps {
  className?: string;
}

export const IconObstacleAlert = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
  </svg>
);

export const IconGpsNavigation = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z" />
  </svg>
);

export const IconMobileApp = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3" />
  </svg>
);

export const IconSupportNetwork = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-3.741-5.757M12 12.75L12 15m0 0l-2.25-2.25M12 15l2.25-2.25M12 12.75l-2.25 2.25M12 12.75l2.25 2.25M3 18.72C3.309 18.374 3.702 18 4.156 18m-.404 2.456A9.003 9.003 0 0112 3c4.236 0 7.92 2.706 8.844 6.544M12 3v1.875m0 0H9.375M12 4.875L14.625 4.875M12 3S7.072 2.25 4.156 4.125c-2.072 1.306-3.587 3.994-3.08 6.88a9.037 9.037 0 003.08 4.5H4.156" />
  </svg>
);


export const IconAutonomy = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
  </svg>
);

export const IconSafety = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
  </svg>
);

export const IconQualityOfLife = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" />
  </svg>
);

export const IconInnovation = ({ className = "w-6 h-6" }: IconProps): React.ReactNode => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.354a15.054 15.054 0 01-4.5 0M12 6.75h.008v.008H12V6.75zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
  </svg>
);

export const SENAI_LOGO_SVG = `data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNTYiIGhlaWdodD0iMjU2IiB2aWV3Qm94PSIwIDAgNzIgNzIiPjxwYXRoIGZpbGw9IiMwMDVFN0IiIGQ9Ik0zNiAwQzE2LjEyIDAgMCAxNi4xMiAwIDM2czE2LjEyIDM2IDM2IDM2IDM2LTE2LjEyIDM2LTM2UzU1LjkgMCAzNiAwem0yMC4yIDEzLjhjLTEuNyAyLjEtMy42IDMuOC01LjggNC45bC0uMy0xLjljLjEtLjEuMS0uMy4xLS40IDAtMS0uMy0xLjgtMS0yLjItLjctLjUtMS42LS42LTIuNS0uNnYtMi4xaC0xLjF2Mi4xYy0uOSAwLTEuOC4xLTIuNy40di0yLjVoLTEuMXYyLjRjLS41IDAtMS4xLjItMS41LjR2LTIuOGgtMS4xdjIuN2MtMS4xLjQtMS45IDEuMS0yLjIgMi4xLS4yLjYtLjIgMS4yIDAgMS44bDEuMy41di4xaC0xLjRsLjMgMS4xdi4xYzAgMS40LjcgMi42IDIuMSAzLjIgMS44LjggMy45LjggNS42LjJsLjctMS4zYy0uMSAwLS4xLS4xLS4xLS4ycy4xLS4yLjEtLjNjLjQtLjQuNC0xIC4xLTEuNC0uMy0uNC0xLjEtLjgtMi0uOC0uOS0uMS0xLjYgMC0yLjIuNC0uMS4xLS4yLjEtLjIuM2wtMS0uM3YtLjFjMC0uMyAwLS41LjEtLjhzLjQtLjUuOC0uN2MuOS0uMyAxLjgtLjUgMi44LS41aC4xdjJoMS4xdjJoMS4xdjJoMS4xdjJoMS4xbDIuMi4xYzEuMi4xIDIuNC0uMSAzLjUtLjggMS0uNiAxLjYtMS41IDEuNy0yLjhsLjEtMy43Yy4xLTEuMi0uMi0yLjQtLjctMy40em0tMzYgMzguNEMyMS42IDUyLjIgMTQuMyA0Ny4zIDkuOCA0MC42Yy40LS41LjktMSAxLjMtMS40IDQuMi00LjEgOS03LjQgMTMuOS0xMCAxLjUtLjggMy0xLjUgNC42LTIuMS41LS4yIDEALjMgMS41LS40IDMuMi0xLjQgNi4xLTMuMyA4LjQtNS45LTEuMi4xLTIuMy4zLTMuNC42LTQgMS04IDIgMTEuNyAzLjYtMi40IDMtNS45IDQuNy05LjcgNS42LTMuOS45LTcuNyAxLjEtMTEuMy43LTU-.5LTkuNC0xLjgtMTIuOC00LjQtMS4yLTEtMi4yLTIuMS0zLjEtMy40LTEuMyAxLjItMi41IDIuNS0zLjYgMy44LTEuMiAxLjMtMi4zIDIuNi0zLjQgMy45LS41LjctMSAxLjQtMS4zIDIuMi0xLjQgMS4zLTIuOCAyLjctNC4xIDQuMS0uNy43LTEuMyAxLjUtMS44IDIuMi0uMy41LS41IDEuMS0uNiAxLjYtLjggMy0uNyA2LjIgMS4xIDguNyAyLjIgMy4xIDUuOCA0LjUgOS40IDQuNCAyLjYtLjEgNS4xLS43IDcuNC0xLjkgMS4yLS42IDIuMy0xLjQgMy4zLTIuM2wtMi42LTQuMmMtLjYtMS0uMy0yLjMgMS0yLjcgMS0uMyAyLjEtLjEgMi44LjcgMS4yIDEuNy43IDQuMS0xLjIgNS41LTEuMS44LTIuMyAxLjQtMy42IDEuNy0yLjgtLjQtNS4xLTEuNS02LjgtMy4zLTEuNS0xLjYtMi4zLTMuNy0yLTUuOC4yLTEuMy43LTIuNSAxLjQtMy42cy4zLTEuMS4zLTEuN2MwIDAgLjEtLjEuMS0uMWwtNC4xLTIuNmMtMS41IDEuOC0yLjIgMy45LTEuOSA2LjEuMyAyLjkgMS45IDUuMyA0LjEgNi42IDEuMy44IDIuOCAxLjIgNC4zIDEuMiAxLjUgMCAzLS40IDQuMy0xLjIgMi4zLTEuNCAzLjYtMy44IDMuNS02LjUtLjEtMy4zLTEuOS02LjEtNC41LTcuNmgtLjFjLS40LS4yLS44LS40LTEuMi0uNy0xLjQtLjgtMi41LTIuMS0zLjEtMy41LS44LTEuOC0uOS0zLjggMC01LjVsNC4yIDIuNmဒီMC45IDEuNCAxLjQgMy4xIDEuMiA0LjktLjUgMS44LTEuNyAzLTEzLjIgMy41LTEuMi4zLTIuMS45LTIuNiAxLjdsMy4yIDUuNGMxLjYtLjcgMi45LTEuNyAzLjctMy4xWiIvPjwvc3ZnPg==`;
