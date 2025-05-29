
import React from 'react';

export type Language = 'en' | 'pt' | 'es';

export interface Feature {
  id: string;
  icon?: React.ReactElement<{ className?: string }>;
  titleKey: string; // Translatable title
  descriptionKey: string; // Translatable description
  imagePlaceholderClass?: string;
  linkUrl?: string;
}

export interface Benefit {
  id: string;
  icon?: React.ReactElement<{ className?: string }>;
  titleKey: string; // Translatable title
  textKey: string; // Translatable text
}

export interface AIModelDetail {
  id: string;
  nameKey: string; // Translatable name
  type: 'Online' | 'Offline' | 'Proprietary';
  descriptionKey: string; // Translatable description
  keyFeaturesKeys?: string[]; // Array of translation keys for features
  logoComponent?: React.ReactNode;
  bgColorClass?: string;
  textColorClass?: string;
}

// For NavItems in Header/MobileNav
export interface NavItemLink {
  href: string;
  labelKey: string; // Translation key for the label
  subItems?: NavItemLink[];
}
