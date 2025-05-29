
import React from 'react';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { WhatIsTrackieSection } from './components/WhatIsTrackieSection';
import { ProductModalitiesSection } from './components/ProductModalitiesSection';
import { BenefitsSection } from './components/BenefitsSection'; 
import { HatConceptSection } from './components/HatConceptSection'; 
import { SpotWaySection } from './components/SpotWaySection'; 
import { RaspWaySection } from './components/RaspWaySection'; 
import { TrackieMobileSection } from './components/TrackieMobileSection'; 
import { FeaturesSection } from './components/FeaturesSection';
import { AIEngineSection } from './components/AIEngineSection';
import { VisionSection } from './components/VisionSection';
import { Footer } from './components/Footer';
import { LanguageProvider } from './i18n/LanguageContext';

const AppContent: React.FC = () => {
  return (
    <div className="min-h-screen overflow-x-hidden bg-brand-bg text-brand-primary-text bg-noise selection:bg-brand-accent-glow selection:text-brand-bg">
      <Header />
      <main className="relative z-[1]">
        <HeroSection />
        <WhatIsTrackieSection />
        <ProductModalitiesSection />
        <BenefitsSection />      
        <HatConceptSection />      
        <SpotWaySection />         
        <RaspWaySection />         
        <TrackieMobileSection />   
        <FeaturesSection />
        <AIEngineSection />
        <VisionSection />
      </main>
      <Footer />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
};

export default App;
