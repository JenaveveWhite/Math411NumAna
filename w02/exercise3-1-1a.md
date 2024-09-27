# Exercise 3.1.1a (C3-P1)

## Use Lagrange interpolation to find a polynomial that passes through the points $(0, 1), (2, 3), (3, 0)$.

The Lagrange interpolation polynomial for three points $(x_1, y_1), (x_2, y_2),$ and $(x_3, y_3)$ is given by the formula:

$$P(x) = y_1 \frac{(x - x_2)(x - x_3)}{(x_1 - x_2)(x_1 - x_3)} + y_2 \frac{(x - x_1)(x - x_3)}{(x_2 - x_1)(x_2 - x_3)} + y_3 \frac{(x - x_1)(x - x_2)}{(x_3 - x_1)(x_3 - x_2)}$$

The points are:

- $(x_1, y_1) = (0, 1)$
- $(x_2, y_2) = (2, 3)$
- $(x_3, y_3) = (3, 0)$

### Step-by-step calculation:

1. **First term** (corresponding to $(x_1, y_1) = (0, 1)$):

$$
1 \cdot \frac{(x - 2)(x - 3)}{(0 - 2)(0 - 3)} = 1 \cdot \frac{(x - 2)(x - 3)}{(-2)(-3)} = \frac{(x - 2)(x - 3)}{6}
$$

2. **Second term** (corresponding to $(x_2, y_2) = (2, 3)$)

$$
3 \cdot \frac{(x - 0)(x - 3)}{(2 - 0)(2 - 3)} = 3 \cdot \frac{(x)(x - 3)}{(2)(-1)} = -\frac{3x(x - 3)}{2}
$$

3. **Third term** (corresponding to $(x_3, y_3) = (3, 0)$):

$$
0 \cdot \frac{(x - 0)(x - 2)}{(3 - 0)(3 - 2)} = 0
$$

### Combine the terms:

$$
P(x) = \frac{(x - 2)(x - 3)}{6} - \frac{3x(x - 3)}{2}
$$

### Simplify:

First term:

$$
\frac{(x - 2)(x - 3)}{6} = \frac{x^2 - 5x + 6}{6}
$$

Second term:

$$
-\frac{3x(x - 3)}{2} = -\frac{3(x^2 - 3x)}{2} = -\frac{3x^2}{2} + \frac{9x}{2}
$$

Now, combine these two terms:

$$
P(x) = \frac{x^2 - 5x + 6}{6} - \left(\frac{3x^2}{2} - \frac{9x}{2}\right)
$$

To combine, first rewrite everything with a denominator of 6:

$$
P(x) = \frac{x^2 - 5x + 6}{6} - \frac{9x^2 - 27x}{6}
$$

Now simplify:

$$
P(x) = \frac{x^2 - 5x + 6 - 9x^2 + 27x}{6}
$$

$$
P(x) = \frac{-8x^2 + 22x + 6}{6}
$$

This is the final polynomial:

$$
P(x) = \frac{-4x^2 + 11x + 3}{3}
$$

This is the interpolating polynomial that passes through the points $(0, 1)$, $(2, 3)$, and $(3, 0)$.
