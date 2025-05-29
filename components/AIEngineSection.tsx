
import React from 'react';
import { SectionTitle } from './SectionTitle';
import AnimatedElement from './AnimatedElement';
// AIModelDetailType is not used directly for meta arrays anymore, but AIModelCardPassedProps aligns with parts of it
import { useLanguage } from '../i18n/LanguageContext';

// --- START LOGO COMPONENTS ---
const openai_logo_url = '/Logosymbols/CHATGPT-GREEN.jpg'; 
const deepseek_logo_url = '/Logosymbols/DEEPSEEK.png'; 
const meta_logo_url = '/Logosymbols/LLAMA.png'; 
const google_gemini_logo_url = '/Logosymbols/GEMINI.png'; 
const google_gemma_logo_url = '/Logosymbols/GEMMA.png';
const julius_kaiser_logo_url = '/Logosymbols/julius_kaiser_logo.png';
const spotway_logo_url = '/Logosymbols/SPOTWAY.png';
const raspway_logo_url = '/Logosymbols/RASPWAY.png';

interface LogoProps {
  className?: string;
  altText: string; // Will receive translated alt text
}

const LogoImage: React.FC<LogoProps & { src: string }> = ({ src, altText, className = "h-10 w-auto object-contain" }) => (
  <img src={src} alt={altText} className={`${className} transition-transform duration-300 group-hover:scale-105`} loading="lazy"/>
);

// Logo components now expect translated alt text via props, passed down from AIEngineSection
const LogoOpenAI: React.FC<Omit<LogoProps, 'altText'> & { altText: string }> = (props) => <LogoImage src={openai_logo_url} {...props} />;
const LogoDeepSeek: React.FC<Omit<LogoProps, 'altText'> & { altText: string }> = (props) => <LogoImage src={deepseek_logo_url} {...props} />;
const LogoMeta: React.FC<Omit<LogoProps, 'altText'> & { altText: string }> = (props) => <LogoImage src={meta_logo_url} {...props} />;
const LogoGoogleAI: React.FC<Omit<LogoProps, 'altText'> & { altText: string }> = (props) => <LogoImage src={google_gemini_logo_url} {...props} />;
const LogoGoogleGemma: React.FC<Omit<LogoProps, 'altText'> & { altText: string }> = (props) => <LogoImage src={google_gemma_logo_url} {...props} />;

const LogoNamoSSLM: React.FC<Omit<LogoProps, 'altText'> & { altText: string, className?: string }> = ({ className = "h-10 md:h-12 w-auto text-brand-accent-subtle", altText }) => (
  <svg viewBox="0 0 100 40" className={className} fill="currentColor" aria-label={altText}>
    <title>{altText}</title>
    <text x="5" y="30" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold">
      NAMO
      <tspan fillOpacity="0.7" fontSize="16" dy="-0.2em">sslm</tspan>
    </text>
  </svg>
);

const LogoJuliusKaiser: React.FC<Omit<LogoProps, 'altText'> & { altText: string, className?: string }> = ({ className = "h-20 w-20 md:h-28 md:w-28 object-contain", altText }) => (
  <LogoImage src={julius_kaiser_logo_url} altText={altText} className={className} />
);

export const LogoSpotWay: React.FC<Omit<LogoProps, 'altText'> & { altText: string, className?: string }> = ({ className = "h-12 md:h-14 w-auto object-contain", altText }) => (
  <LogoImage src={spotway_logo_url} altText={altText} className={className} />
);
export const LogoRaspWay: React.FC<Omit<LogoProps, 'altText'> & { altText: string, className?: string }> = ({ className = "h-12 md:h-14 w-auto object-contain", altText }) => (
  <LogoImage src={raspway_logo_url} altText={altText} className={className} />
);
// --- END LOGO COMPONENTS ---

// Type for the props AIModelCard expects after processing/translation
interface AIModelCardPassedProps {
    id: string; // For keying
    name: string;
    type: 'Online' | 'Offline' | 'Proprietary';
    description: string;
    keyFeatures?: string[];
    logoComponent?: React.ReactNode;
    bgColorClass?: string;
    textColorClass?: string;
}

// Type for the raw metadata defined in the component for online/offline models
interface AIModelSourceMeta {
  id: string;
  type: 'Online' | 'Offline' | 'Proprietary';
  logoComponentProvider: (alt: string) => React.ReactNode;
  bgColorClass?: string;
  textColorClass?: string;
}

// Type for JuliusKaiser raw metadata (different structure)
interface JuliusKaiserSourceMeta {
    id: string;
    type: 'Proprietary';
    bgColorClass?: string;
    textColorClass?: string;
    // No logoComponentProvider, logo is handled directly
}


const AIModelCard: React.FC<{ model: AIModelCardPassedProps, index: number }> = ({ model, index }) => {
  const { t } = useLanguage();
  return (
    <AnimatedElement
      initialClasses="opacity-0 translate-y-10"
      finalClasses="opacity-100 translate-y-0"
      transitionClasses="transition-all duration-700 ease-out"
      style={{transitionDelay: `${index * 150 + 300}ms`}}
      className={`group p-6 md:p-7 rounded-xl shadow-xl h-full flex flex-col 
                  ${model.bgColorClass || 'bg-brand-surface/70 backdrop-blur-lg'} 
                  ${model.textColorClass || 'text-brand-primary-text'} 
                  border border-brand-surface-alt/60 hover:border-brand-accent-glow/50 
                  hover:shadow-brand-accent-glow/20 transition-all duration-300 transform hover:-translate-y-1`}
    >
      {model.logoComponent && <div className="mb-5 h-10 md:h-12 flex items-center justify-start">{model.logoComponent}</div>}
      <h4 className="text-xl md:text-2xl font-bold mb-3 flex items-center group-hover:text-brand-accent-highlight transition-colors duration-200">
        {model.name} 
        {model.type === 'Proprietary' && <span className="ml-2.5 px-2.5 py-1 text-xs font-semibold bg-brand-accent-glow/20 text-brand-accent-highlight rounded-full uppercase tracking-wide">{t('aiEngineSection.modelTypes.proprietary')}</span>}
        {model.type === 'Online' && <span className="ml-2.5 px-2.5 py-1 text-xs font-medium bg-sky-500/20 text-sky-300 rounded-full">{t('aiEngineSection.modelTypes.online')}</span>}
        {model.type === 'Offline' && <span className="ml-2.5 px-2.5 py-1 text-xs font-medium bg-emerald-500/20 text-emerald-300 rounded-full">{t('aiEngineSection.modelTypes.offline')}</span>}
      </h4>
      <p className={`text-sm md:text-[15px] mb-5 leading-relaxed ${model.textColorClass || 'text-brand-secondary-text'}`}>{model.description}</p>
      {model.keyFeatures && model.keyFeatures.length > 0 && (
        <ul className="list-disc list-inside space-y-1.5 text-sm mt-auto pl-1">
          {model.keyFeatures.map((feature, idx) => (
            <li key={idx} className={`${model.textColorClass || 'text-brand-secondary-text'}`}>{feature}</li>
          ))}
        </ul>
      )}
    </AnimatedElement>
  );
};


const TextPopInAnimator: React.FC<{ text: string; className?: string; charDelay?: number }> = ({ text, className, charDelay = 0.03 }) => {
  return (
    <span className={`inline-block ${className}`}>
      {text.split('').map((char, index) => (
        <span key={index} className="inline-block opacity-0 animate-textPopInChar" style={{ animationDelay: `${index * charDelay}s` }}>
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </span>
  );
};


export const AIEngineSection: React.FC = () => {
  const { t, tArray } = useLanguage();

  const onlineModelsMeta: AIModelSourceMeta[] = [
    { id: 'gemini', type: 'Online', logoComponentProvider: (alt: string) => <LogoGoogleAI className="h-10 md:h-12" altText={alt}/>, bgColorClass: 'bg-gradient-to-br from-sky-800/50 to-blue-900/50' },
    { id: 'openai-realtime', type: 'Online', logoComponentProvider: (alt: string) => <LogoOpenAI className="h-10 md:h-12" altText={alt}/>, bgColorClass: 'bg-gradient-to-br from-emerald-800/50 to-green-900/50' },
    { id: 'namo-sslm', type: 'Online', logoComponentProvider: (alt: string) => <LogoNamoSSLM altText={alt}/>, bgColorClass: 'bg-gradient-to-br from-purple-800/50 to-indigo-900/50' },
  ];

  const offlineModelsMeta: AIModelSourceMeta[] = [
    { id: 'gemma', type: 'Offline', logoComponentProvider: (alt: string) => <LogoGoogleGemma className="h-10 md:h-12" altText={alt}/>, bgColorClass: 'bg-brand-surface-alt/80' },
    { id: 'deepseek', type: 'Offline', logoComponentProvider: (alt: string) => <LogoDeepSeek className="h-10 md:h-12" altText={alt}/>, bgColorClass: 'bg-brand-surface-alt/80' },
    { id: 'llama', type: 'Offline', logoComponentProvider: (alt: string) => <LogoMeta className="h-10 md:h-12" altText={alt}/>, bgColorClass: 'bg-brand-surface-alt/80' },
  ];
  
  const juliusKaiserMeta: JuliusKaiserSourceMeta = { // Uses the specific meta type for JuliusKaiser
    id: 'juliuskaiser', type: 'Proprietary', bgColorClass: 'bg-transparent', textColorClass: 'text-brand-primary-text',
  };

  const mapModelData = (meta: AIModelSourceMeta, modelKey: string): AIModelCardPassedProps => ({
    id: meta.id,
    type: meta.type,
    bgColorClass: meta.bgColorClass,
    textColorClass: meta.textColorClass,
    name: t(`aiEngineSection.models.${modelKey}.name`),
    description: t(`aiEngineSection.models.${modelKey}.description`),
    keyFeatures: tArray(`aiEngineSection.models.${modelKey}.features`),
    logoComponent: meta.logoComponentProvider(t(`aiEngineSection.logoAlt.${modelKey}`)),
  });
  
  const onlineModelsData = onlineModelsMeta.map(meta => mapModelData(meta, meta.id));
  const offlineModelsData = offlineModelsMeta.map(meta => mapModelData(meta, meta.id));
  
  // JuliusKaiser data is constructed slightly differently as it doesn't use logoComponentProvider
  const juliusKaiserData: AIModelCardPassedProps = {
      id: juliusKaiserMeta.id,
      type: juliusKaiserMeta.type,
      bgColorClass: juliusKaiserMeta.bgColorClass,
      textColorClass: juliusKaiserMeta.textColorClass,
      name: t(`aiEngineSection.models.julius.name`),
      description: t(`aiEngineSection.models.julius.description`),
      keyFeatures: tArray(`aiEngineSection.models.julius.features`),
      // logoComponent for JuliusKaiser is directly rendered in JSX, not part of this data object for the card
  };


  return (
    <section id="ai-engine" className="py-28 md:py-40 bg-brand-bg content-over-noise"> 
      <div className="container mx-auto px-6 lg:px-8">
        <SectionTitle
          subtitle={t('aiEngineSection.subtitle')}
          title={t('aiEngineSection.title')}
          description={t('aiEngineSection.description')}
        />

        <AnimatedElement initialClasses="opacity-0" finalClasses="opacity-100" transitionClasses="transition-opacity duration-1000 ease-in delay-300ms">
          <h3 className="text-3xl md:text-4xl font-bold text-brand-primary-text mt-16 mb-12 text-center md:text-left" style={{textShadow: '0 0 12px rgba(var(--color-brand-accent-glow-rgb), 0.4)'}}>
            {t('aiEngineSection.onlineTitle')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 md:gap-10 mb-20">
            {onlineModelsData.map((model, index) => (
              <AIModelCard key={model.id} model={model} index={index} />
            ))}
          </div>
        </AnimatedElement>

        <AnimatedElement initialClasses="opacity-0" finalClasses="opacity-100" transitionClasses="transition-opacity duration-1000 ease-in delay-500ms">
          <h3 className="text-3xl md:text-4xl font-bold text-brand-primary-text mt-16 mb-12 text-center md:text-left" style={{textShadow: '0 0 12px rgba(var(--color-brand-accent-subtle-rgb), 0.4)'}}>
            {t('aiEngineSection.offlineTitle')}
          </h3>
          <p className="text-brand-secondary-text text-base md:text-lg mb-12 text-center md:text-left max-w-3xl mx-auto md:mx-0">
            {t('aiEngineSection.offlineDescription')}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 md:gap-10 mb-24">
            {offlineModelsData.map((model, index) => (
              <AIModelCard key={model.id} model={model} index={index} />
            ))}
          </div>
        </AnimatedElement>
        
        <AnimatedElement 
            initialClasses="opacity-0 scale-95" 
            finalClasses="opacity-100 scale-100" 
            transitionClasses="transition-all duration-1000 ease-out delay-[700ms]" 
            className="my-20 md:my-28"
        >
          <div 
            className="relative p-8 md:p-12 lg:p-16 rounded-2xl overflow-hidden border border-brand-accent-glow/30 shadow-2xl shadow-brand-accent-glow/30 
                       bg-gradient-to-br from-brand-surface via-brand-surface-alt to-brand-surface animate-gradient-wave" 
            style={{ '--tw-gradient-from': 'rgba(var(--color-brand-accent-glow-rgb), 0.1)', '--tw-gradient-to': 'rgba(var(--color-brand-accent-highlight-rgb),0.05)' } as React.CSSProperties}
          >
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'80\' height=\'80\' viewBox=\'0 0 80 80\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%23FFF\' fill-opacity=\'1\'%3E%3Cpath d=\'M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'}}></div>
            
            <div className="relative z-10 text-center">
              <AnimatedElement
                initialClasses="opacity-0 scale-75"
                finalClasses="opacity-100 scale-100"
                transitionClasses="transition-all duration-1000 ease-out delay-[900ms]"
                className="mb-8 md:mb-10 flex justify-center"
              >
                <LogoJuliusKaiser altText={t('aiEngineSection.logoAlt.juliusKaiser')} />
              </AnimatedElement>

              <div className="flex items-center justify-center mb-6 md:mb-8">
                <h3 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight">
                  <TextPopInAnimator text={juliusKaiserData.name} className="text-brand-primary-text text-shadow-glow-accent-md" charDelay={0.05}/>
                </h3>
                <AnimatedElement
                  initialClasses="opacity-0 scale-75 -translate-x-2"
                  finalClasses="opacity-100 scale-100 translate-x-0"
                  transitionClasses="transition-all duration-700 ease-out delay-[1600ms]" 
                >
                  <span 
                    className="ml-3 sm:ml-4 px-3.5 py-1.5 text-xs sm:text-sm font-bold uppercase tracking-wider bg-gradient-to-r from-yellow-400 to-amber-500 text-yellow-900 rounded-lg shadow-lg transform -rotate-2 animate-subtle-glow-pulse"
                    style={{ "--color-brand-accent-glow-rgb": "var(--color-yellow-glow-rgb)" } as React.CSSProperties} 
                  >
                    {t('aiEngineSection.proprietarySubtitle')}
                  </span>
                </AnimatedElement>
              </div>

              <p className="text-lg md:text-xl text-brand-secondary-text mb-10 max-w-3xl mx-auto leading-relaxed" style={{textShadow: '0 0 5px rgba(0,0,0,0.7)'}}>
                {juliusKaiserData.description}
              </p>
              <div className="mt-6 mb-10">
                <span className="inline-block bg-brand-surface text-brand-accent-subtle text-sm font-medium px-5 py-2.5 rounded-lg shadow-md hover:bg-brand-surface-alt transition-colors border border-brand-surface-alt">
                  {t('aiEngineSection.developedAt')}
                </span>
              </div>
              {juliusKaiserData.keyFeatures && (
                <div className="mb-12 flex flex-wrap justify-center gap-3 md:gap-4">
                  {juliusKaiserData.keyFeatures.map((feature, idx) => (
                     <AnimatedElement 
                        key={feature}
                        initialClasses="opacity-0 scale-80"
                        finalClasses="opacity-100 scale-100"
                        transitionClasses="transition-all duration-500 ease-out"
                        style={{transitionDelay: `${2000 + idx * 100}ms`}}
                      >
                        <span className="px-4 py-2 text-xs md:text-sm font-medium bg-brand-accent-glow/10 text-brand-accent-highlight rounded-full border border-brand-accent-glow/30 hover:bg-brand-accent-glow/20 hover:text-white transition-all cursor-default shadow-sm">
                          {feature}
                        </span>
                    </AnimatedElement>
                  ))}
                </div>
              )}
              <div className="mt-12 text-center">
                <AnimatedElement
                  initialClasses="opacity-0 scale-90"
                  finalClasses="opacity-100 scale-100"
                  transitionClasses="transition-all duration-1000 ease-out delay-[2500ms]" 
                >
                  <button
                    onClick={(e) => e.preventDefault()} 
                    className="group relative inline-flex items-center justify-center px-8 py-4 text-base md:text-lg font-semibold text-brand-primary-text rounded-xl overflow-hidden
                               bg-gradient-to-r from-brand-accent-subtle to-brand-accent-glow 
                               border-2 border-transparent text-brand-bg
                               hover:from-brand-accent-glow hover:to-brand-accent-highlight
                               transition-all duration-300 ease-out
                               focus:outline-none focus-visible:ring-4 focus-visible:ring-brand-accent-glow/50
                               transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-brand-accent-glow/50"
                    aria-label={t('aiEngineSection.knowMoreJulius')}
                  >
                    <span className="shine-effect absolute inset-0"></span>
                    <span 
                      className="absolute inset-0 border-2 border-transparent rounded-xl 
                                 group-hover:border-white/30 
                                 transition-all duration-300 opacity-0 group-hover:opacity-100">
                    </span>
                    <span className="relative z-10 flex items-center">
                      <svg 
                        className="w-6 h-6 mr-3 transform transition-all duration-500 ease-out 
                                   group-hover:rotate-[360deg] group-hover:scale-125" 
                        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L1.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.25 12L17 13.75M17 13.75L15.75 12M17 13.75L18.25 15.5M17 13.75L19.25 13.75" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75C12.9957 18.75 13.9481 18.5568 14.7883 18.1918C15.6286 17.8268 16.3299 17.3016 16.9646 16.6669C17.5993 16.0322 18.1244 15.3309 18.4895 14.5907C18.8545 13.7504 19.0477 12.898 19.0477 11.9023C19.0477 10.9066 18.8545 10.0542 18.4895 9.21399C18.1244 8.47375 17.5993 7.77248 16.9646 7.13781C16.3299 6.50314 15.6286 5.97796 14.7883 5.61298C13.9481 5.248 12.9957 5.05481 12 5.05481C11.0043 5.05481 10.0519 5.248 9.21168 5.61298C8.37144 5.97796 7.67017 6.50314 7.0355 7.13781C6.40083 7.77248 5.87565 8.47375 5.51067 9.21399C5.14569 10.0542 4.9525 10.9066 4.9525 11.9023C4.9525 12.898 5.14569 13.7504 5.51067 14.5907C5.87565 15.3309 6.40083 16.0322 7.0355 16.6669C7.67017 17.3016 8.37144 17.8268 9.21168 18.1918C10.0519 18.5568 11.0043 18.75 12 18.75Z" />
                      </svg>
                      {t('aiEngineSection.waitNews')}
                    </span>
                  </button>
                </AnimatedElement>
              </div>
            </div>
          </div>
        </AnimatedElement>

      </div>
    </section>
  );
};
