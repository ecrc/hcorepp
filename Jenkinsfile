pipeline {
    agent { label 'Vulture' }
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -le
                             ####################################################
                            # Configure and build
                            ####################################################
                            echo "========================================"
                            # load cmake module for the build
                            module load cmake-3.22.1-gcc-7.5.0-4se4k5d
                            mkdir -p build
                            cd build && rm -rf ./*
                            pwd
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
                            cd tests
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
