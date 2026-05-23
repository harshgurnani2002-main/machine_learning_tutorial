const fs = require('fs');
const content = fs.readFileSync('src/data/modules/gradientBoosting.ts', 'utf8');
// We can try to require it using ts-node or just find the syntax error using acorn
