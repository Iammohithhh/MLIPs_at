import { mulberry32, randomRotation } from '../chem/rng';
import type { Mat3 } from './escn';

export { axisRotation } from './escn';

export const randomRotationMatrix = (seed: number): Mat3 => randomRotation(mulberry32(seed));
