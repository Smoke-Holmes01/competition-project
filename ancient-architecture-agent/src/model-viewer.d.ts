import type {DetailedHTMLProps, HTMLAttributes} from 'react';

type ModelViewerProps = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
  src?: string;
  poster?: string;
  alt?: string;
  'camera-controls'?: boolean | string;
  'auto-rotate'?: boolean | string;
  'shadow-intensity'?: string;
  exposure?: string;
  'environment-image'?: string;
  'interaction-prompt'?: string;
  onLoad?: (event: Event) => void;
};

export type ModelViewerElement = HTMLElement;

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': ModelViewerProps;
    }
  }
}
