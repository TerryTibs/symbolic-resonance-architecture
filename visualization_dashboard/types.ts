import React from 'react';

export enum ModuleStatus {
  IMPLEMENTABLE_TODAY = "Implementable Today",
  ADVANCED_PROTOTYPE = "Advanced Prototype",
  RESEARCH_PROTOTYPE = "Research Prototype",
  CORE_INNOVATION = "Core Innovation",
}

export interface SraModule {
  name: string;
  acronym: string;
  description: string;
  existingTech: string;
  novelAspect: string;
  status: ModuleStatus;
  // Fix: Changed from JSX.Element to React.ReactElement to explicitly use the React import and avoid potential 'JSX' namespace issues.
  icon: React.ReactElement;
}

export interface RoadmapStageData {
  stage: number;
  title: string;
  description: string;
  modules: SraModule[];
}
