
import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import { translations } from './translations';
import { Language } from '../types';

type TranslationKey = string | string[];

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: TranslationKey, replacements?: Record<string, string>) => string;
  tArray: (key: TranslationKey) => string[];
  tObj: (key: TranslationKey) => any; // For fetching nested objects
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const getInitialLanguage = (): Language => {
  if (typeof window !== 'undefined') {
    const savedLang = localStorage.getItem('trackway-lang');
    if (savedLang && (savedLang === 'en' || savedLang === 'pt' || savedLang === 'es')) {
      return savedLang as Language;
    }
    const browserLang = navigator.language.split('-')[0];
    if (browserLang === 'en' || browserLang === 'pt' || browserLang === 'es') {
      return browserLang as Language;
    }
  }
  return 'pt'; // Default language
};

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(getInitialLanguage);

  useEffect(() => {
    if (typeof window !== 'undefined') {
        localStorage.setItem('trackway-lang', language);
    }
  }, [language]);

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
  };

  const resolveKey = (key: TranslationKey, lang: Language): any => {
    const keyParts = Array.isArray(key) ? key : key.split('.');
    let current: any = translations[lang];
    for (const part of keyParts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        return undefined; // Key not found
      }
    }
    return current;
  };
  
  const t = (key: TranslationKey, replacements?: Record<string, string>): string => {
    let value = resolveKey(key, language);

    if (value === undefined) {
      // Fallback to English if key not found in current language, then to default key
      value = resolveKey(key, 'en');
      if (value === undefined) {
        console.warn(`Translation key "${Array.isArray(key) ? key.join('.') : key}" not found in language "${language}" or fallback "en".`);
        return Array.isArray(key) ? key.join('.') : key;
      }
    }
    
    if (typeof value !== 'string') {
        console.warn(`Translation for key "${Array.isArray(key) ? key.join('.') : key}" is not a string for language "${language}".`);
        return Array.isArray(key) ? key.join('.') : key;
    }

    if (replacements) {
      return Object.entries(replacements).reduce((acc, [placeholder, replacementValue]) => {
        return acc.replace(new RegExp(`{${placeholder}}`, 'g'), replacementValue);
      }, value);
    }
    return value;
  };

  const tArray = (key: TranslationKey): string[] => {
    let value = resolveKey(key, language);
     if (value === undefined) {
      value = resolveKey(key, 'en');
       if (value === undefined) {
        console.warn(`Translation array key "${Array.isArray(key) ? key.join('.') : key}" not found in language "${language}" or fallback "en".`);
        return [];
      }
    }
    if (!Array.isArray(value) || !value.every(item => typeof item === 'string')) {
        console.warn(`Translation for array key "${Array.isArray(key) ? key.join('.') : key}" is not an array of strings for language "${language}".`);
        return [];
    }
    return value;
  };

  const tObj = (key: TranslationKey): any => {
    let value = resolveKey(key, language);
    if (value === undefined) {
      value = resolveKey(key, 'en');
      if (value === undefined) {
        console.warn(`Translation object key "${Array.isArray(key) ? key.join('.') : key}" not found in language "${language}" or fallback "en".`);
        return {};
      }
    }
    return value;
  };


  return (
    <LanguageContext.Provider value={{ language, setLanguage, t, tArray, tObj }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = (): LanguageContextType => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
