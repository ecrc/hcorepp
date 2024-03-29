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
        stage ('mkl') {
            agent { label 'jenkinsfile' }
            stages {
                stage ('build') {
                    steps {
                        sh '''#!/bin/bash -le
                            ####################################################
                            # Configure and build
                            ####################################################
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166
                            ./config.sh -t -e
                            ./clean_build.sh
                        '''
                    }
                }
                stage ('test') {
                    steps {

                        sh '''#!/bin/bash -le
                            ####################################################
                            # Run tester
                            ####################################################
                            echo "========================================"
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
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
        }
        stage('openblas') {
            agent { label 'jenkinsfile' }
            stages {
                stage ('build') {
                    steps {
                        sh '''#!/bin/bash -le
                            ####################################################
                            # Configure and build
                            ####################################################
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            ./config.sh -t -e
                            ./clean_build.sh
                        '''
                    }
                }
                stage ('test') {
                    steps {
                        sh '''#!/bin/bash -le
                            ####################################################
                            # Run tester
                            ####################################################
                            echo "========================================"
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            cd bin/tests
                            ./hcorepp-tests                
                            '''
                    }
                }
            }
        }
	stage('documentation') {
             agent { label 'jenkinsfile'}
             steps {
                 sh '''#!/bin/bash -le
                    module purge
                    module load gcc/10.2.0
                    module load cmake/3.21.2
                    ####################################################
                    # BLAS/LAPACK
                    ####################################################
                    module load mkl/2020.0.166
                    ./config.sh -t -e
                    ./clean_build.sh
                    cd bin
                    make docs
                    '''
                 publishHTML( target: [allowMissing: false, alwaysLinkToLastBuild: false, keepAll: false, reportDir: 'docs/html', reportFiles: 'index.html', reportName: 'Doxygen Documentation', reportTitles: ''] )
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
