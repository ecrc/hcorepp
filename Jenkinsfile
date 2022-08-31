pipeline {
    agent { label 'jenkinsfile' }
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }
   // environment {
     //   HCORECPPDEVDIR= "{$PWD}"
    //}

    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -le
                             ####################################################
                            # Configure and build
                            ####################################################
                            if [ "${host}" = "Vulture" ]; then
                                module load gcc/7.2.0
                                module load cmake/3.17.3
			    else if ["${host} = "Stork"]; then 
				module load cmake-3.22.1-gcc-7.5.0-4se4k5d
				module load gcc/10.2.0
                            else
                                module load gcc/10.2.0
                                module load cmake/3.19.2
                            fi
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166	
                         #   cd $HCORECPPDEVDIR
                            ./config.sh -t
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
                            if [ "${host}" = "Vulture" ]; then
                                module load gcc/7.2.0
                                module load cmake/3.17.3
                            else
                                module load gcc/10.2.0
                                module load cmake/3.19.2
                            fi
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166
                         #   cd $HCORECPPDEVDIR
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
