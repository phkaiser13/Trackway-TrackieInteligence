
import React from 'react';
import { useLanguage } from '../i18n/LanguageContext';
import { Language } from '../types';

interface LanguageOption {
  code: Language;
  label: string;
  flag: string;
}

const languageOptions: LanguageOption[] = [
  { code: 'pt', label: 'Português', flag: '🇧🇷' },
  { code: 'en', label: 'English', flag: '🇺🇸' },
  { code: 'es', label: 'Español', flag: '🇪🇸' },
];

export const LanguageSwitcher: React.FC<{ onSwitch?: () => void }> = ({ onSwitch }) => {
  const { language, setLanguage, t } = useLanguage();
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  const handleLanguageChange = (langCode: Language) => {
    setLanguage(langCode);
    setIsOpen(false);
    if (onSwitch) onSwitch();
  };
  
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);


  const selectedLanguage = languageOptions.find(opt => opt.code === language) || languageOptions[0];

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center p-2 text-sm font-medium text-brand-secondary-text hover:text-brand-accent-highlight transition-colors duration-200 rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-glow/50"
        aria-haspopup="true"
        aria-expanded={isOpen}
        aria-label={t('header.language')}
      >
        <span className="mr-1.5 text-lg">{selectedLanguage.flag}</span>
        <span className="hidden sm:inline">{selectedLanguage.label}</span>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={`w-4 h-4 ml-1 transition-transform duration-200 ${isOpen ? 'transform rotate-180' : ''}`}>
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.25 4.25a.75.75 0 01-1.06 0L5.23 8.29a.75.75 0 01.02-1.06z" clipRule="evenodd" />
        </svg>
      </button>
      {isOpen && (
        <div 
          className="absolute right-0 mt-2 w-48 bg-brand-surface/95 backdrop-blur-lg shadow-2xl rounded-xl p-2 z-40 border border-brand-surface-alt/50"
          role="menu"
          aria-orientation="vertical"
          aria-labelledby="language-menu-button"
        >
          <ul className="space-y-1" role="none">
            {languageOptions.map((option) => (
              <li key={option.code} role="none">
                <button
                  onClick={() => handleLanguageChange(option.code)}
                  className={`w-full text-left flex items-center px-3.5 py-2.5 text-sm rounded-md transition-all duration-200
                              ${language === option.code 
                                ? 'bg-brand-accent-glow/20 text-brand-accent-highlight font-semibold' 
                                : 'text-brand-secondary-text hover:text-brand-primary-text hover:bg-brand-surface-alt/70'}`}
                  role="menuitem"
                  aria-current={language === option.code ? "true" : "false"}
                >
                  <span className="mr-2.5 text-lg">{option.flag}</span>
                  {option.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
