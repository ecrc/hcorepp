pipeline {
    agent none
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('build_mkl') {
            agent { label 'jenkinsfile' }
            steps {
                sh '''#!/bin/bash -le
                    ####################################################
                    # Configure and build
                    ####################################################
	                module purge
                    module load gcc/10.2.0
	                module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    ####################################################
                    # BLAS/LAPACK
                    ####################################################
                    module load mkl/2020.0.166
                    ./config.sh -t -e
                    ./clean_build.sh
                '''
            }
        }
        stage ('test_mkl') {
            agent { label 'jenkinsfile' }
            steps {

                sh '''#!/bin/bash -le
                    ####################################################
                    # Run tester
                    ####################################################
                    echo "========================================"
                    module purge
		            module load gcc/10.2.0
			        module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    ####################################################
                    # BLAS/LAPACK
                    ####################################################
                    module load mkl/2020.0.166
                    cd bin/tests
                    ./hcorepp-tests                
                    '''
            }
        }
        stage ('build_openblas') {
            agent { label 'jenkinsfile' }
            steps {
                sh '''#!/bin/bash -le
                    ####################################################
                    # Configure and build
                    ####################################################
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    ./config.sh -t -e
                    ./clean_build.sh
                '''
            }
        }
        stage ('test_openblas') {
            agent { label 'jenkinsfile' }
            steps {

                sh '''#!/bin/bash -le
                    ####################################################
                    # Run tester
                    ####################################################
                    echo "========================================"
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    cd bin/tests
                    ./hcorepp-tests                
                    '''
            }
        }
        stage ('build_cuda_openblas') {
            agent { label 'gpu' }
            steps {
                sh '''#!/bin/bash -le
                    ####################################################
                    # Configure and build
                    ####################################################
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    module load cuda/11.6
                    ./config.sh -t -e -c
                    ./clean_build.sh
                '''
            }
        }
        stage ('test_cuda_openblas') {
            agent { label 'gpu' }
            steps {

                sh '''#!/bin/bash -le
                    ####################################################
                    # Run tester
                    ####################################################
                    echo "========================================"
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    module load cuda/11.6
                    cd bin/tests
                    ./hcorepp-tests                
                    '''
            }
        }
        stage ('build_cuda_mkl') {
            agent { label 'gpu' }
            steps {
                sh '''#!/bin/bash -le
                    ####################################################
                    # Configure and build
                    ####################################################
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    module load cuda/11.6
                    ####################################################
                    # BLAS/LAPACK
                    ####################################################
                    module load mkl/2020.0.166
                    ./config.sh -t -e -c
                    ./clean_build.sh
                '''
            }
        }
        stage ('test_cuda_mkl') {
            agent { label 'gpu' }
            steps {

                sh '''#!/bin/bash -le
                    ####################################################
                    # Run tester
                    ####################################################
                    echo "========================================"
                    module purge
                    module load gcc/10.2.0
                    module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                    module load cuda/11.6
                    ####################################################
                    # BLAS/LAPACK
                    ####################################################
                    module load mkl/2020.0.166
                    cd bin/tests
                    ./hcorepp-tests                
                    '''
            }
        }
    }
        
    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
        failure {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
    }
}
