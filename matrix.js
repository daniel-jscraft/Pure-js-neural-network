export class Matrix {

    /**
     * Values are initialized to 0
     * @param {number} Rows
     * @param {number} Columns
     */
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];

        for (let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    multiply(n) {
        let newMatrix = Matrix.multiply(this, n);
        this.data = newMatrix.data;
        this.cols = newMatrix.cols;
        this.rows = newMatrix.rows;
        return this;
    }

    static multiply(matrix1, matrix2) {
        //scalar multiply
        if (matrix1 instanceof Matrix && !(matrix2 instanceof Matrix)) {
            let num = matrix2; //scalar value
            let newMatrix = Matrix.copy(matrix1);
            for (let i = 0; i < matrix1.rows; i++) {
                for (let j = 0; j < matrix1.cols; j++) {
                    newMatrix.data[i][j] *= num;
                }
            }
            return newMatrix;
        }
        if (!(matrix1 instanceof Matrix && matrix2 instanceof Matrix)) {
            throw new Error('Multiplication failed because no Matrix object found!');
        }

        if (matrix2.rows == matrix1.rows && matrix2.cols == matrix1.cols) {
            for (let i = 0; i < matrix1.rows; i++) {
                for (let j = 0; j < matrix1.cols; j++) {
                    matrix1.data[i][j] *= matrix2.data[i][j];
                }
            }
            return matrix1;
        }

        return this.calcMatrixProduct(matrix1, matrix2);
    }

    static calcMatrixProduct(matrix1, matrix2) {
        if (matrix1.cols != matrix2.rows) {
            throw new Error('multiplication failed because of size mismatch!');
        }

        let newMatrix = new Matrix(matrix1.rows, matrix2.cols);
        for (let i = 0; i < newMatrix.rows; i++) {
            for (let j = 0; j < newMatrix.cols; j++) {
                newMatrix.data[i][j] = 0;
                for (let k = 0; k < matrix1.cols; k++) {
                    newMatrix.data[i][j] += matrix1.data[i][k] * matrix2.data[k][j];
                }
            }
        }
        return newMatrix;
    }

    static copy(matrix) {
        let newMat = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                newMat.data[i][j] = matrix.data[i][j];
            }
        }
        return newMat;
    }

   
    print() {
        console.table(this.data);
    }

    static transpose(matrix) { //returns transposed matrix
        let transposed = new Matrix(matrix.cols, matrix.rows);

        for (let i = 0; i < matrix.cols; i++) {
            for (let j = 0; j < matrix.rows; j++) {
                transposed.data[i][j] = matrix.data[j][i];
            }
        }

        return transposed;
    }

    add(matrix) {
        let newMatrix = Matrix.add(this, matrix);
        this.data = newMatrix.data;
        this.cols = newMatrix.cols;
        this.rows = newMatrix.rows;
        return this;
    }

    static add(m1, m2) {
        if (m1 instanceof Matrix && m2 instanceof Matrix) {
            if (m1.rows != m2.rows || m1.cols != m2.cols) {
                throw new Error('invalid addition : size mismatch');
            }
            let newMatrix = new Matrix(m1.rows, m1.cols);
            for (let i = 0; i < m1.rows; i++) {
                for (let j = 0; j < m1.cols; j++) {
                    newMatrix.data[i][j] = m1.data[i][j] + m2.data[i][j];
                }
            }
            return newMatrix;
        }
        for (let i = 0; i < m1.rows; i++) {
            for (let j = 0; j < m1.cols; j++) {
                m1.data[i][j] += m2;
            }
        }
        return m1;
    }

    map(func) {
        let newMatrix = Matrix.map(this, func);
        this.data = newMatrix.data;
        this.cols = newMatrix.cols;
        this.rows = newMatrix.rows;
        return this;
    }

    static map(matrix, func) {
        let newMatrix = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                newMatrix.data[i][j] = func(matrix.data[i][j]);
            }
        }
        return newMatrix;
    }

    static randomize(rows, cols) {
        let newMatrix = new Matrix(rows, cols)
        for (let i = 0; i < newMatrix.rows; i++) {
            for (let j = 0; j < newMatrix.cols; j++) {
                newMatrix.data[i][j] = (Math.random() * 2) - 1; //random number between -1 and 1
            }
        }
        return newMatrix;
    }

    static fromArray(arr) {
        let rows = arr.length;
        let cols = 1;

        let matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            matrix.data[i][0] = arr[i];
        }

        return matrix;
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }
}