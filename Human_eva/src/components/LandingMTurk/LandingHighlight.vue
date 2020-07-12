<template>
    <div class="document">
        <!-- eslint-disable -->
        <h3 align="center">
            Noun Phrases Highlight
        </h3>
        <p>This task requires <strong>three</strong> steps:</p>
        <p><strong>Step 1: Highlighting Informative Phrases in a Document that Support the Bolded Part in a Given Summary the Best</strong></p>
        <p>In this step, you will be given two components <strong>a summary with one bolded noun phrase</strong> and <strong>a document</strong>.</p>
        <p>You should:<br/></p>
            <blockquote>
            1) read the summary and identify the bolded noun phrase <br/>
            2) read the document <br/>
            3) select <strong>Noun Phrases</strong> that support the bolded noun phrase the best<br/>
            </blockquote>
        <p>Examples of highlights:</p>
        
            Example 1:
            <blockquote>
            <strong>Summary:</strong><br/>
            researchers have identified <strong>a gene</strong> that may put people at greater risk of strokes and heart attacks<br/>
            <em>(The noun phrase you should identify is: a gene)</em><br/><br/>
            <strong>Document:</strong><br/>
            ...Writing in PLOS ONE they say the <a class="highlight">gene fault</a> may encourage the formation of blood clots - the ultimate cause of most heart attacks and strokes
            ...<br/>
            Around one in 10 people in the Caucasian population carries <a class="highlight">this variation of the gene , named PIA2</a>
            ...<br/>
            They found individuals with <a class="highlight">PIA2</a> were more likely to have a stroke
            ...<br/>
            the scientists show <a class="highlight">PIA2</a> is also linked to an increased risk of heart attacks in people under 45
            ...<br/>
            </blockquote>

            Example 2:
            <blockquote>            
            <strong>Summary:</strong><br/>
            uk - born oliver hart and bengt holmstrom of finland have won the nobel economics prize <strong>for work on contract theory </strong><br/>
            <em>(The noun phrase you should identify is: for work on contract theory)</em><br/><br/>
            <strong>Document:</strong><br/>
            ...The Royal Swedish Academy of Sciences said that Mr Hart and Mr Holmstrom 's work was " valuable to the understanding of <a class="highlight">real-life contracts and institutions</a> ". 
            ...<br/>
            It also said it could identify " potential pitfalls in <a class="highlight">contract design</a> ."
            ...<br/>
            Mr Holmstrom in particular is known for his research into how <a class="highlight">contracts and incentives</a> affect corporate behaviour. 
            ...<br/>
            </blockquote>

            Example 3:
            <blockquote>            
            <strong>Summary:</strong><br/>
            a ban on growing genetically - modified crops in scotland could threaten <strong>the country 's contribution to scientific research</strong> , according to scientists , universities and farming leaders<br/>
            <em>(The noun phrase you should identify is: the country 's contribution to scientific research)</em><br/><br/>
            <strong>Document:</strong><br/>
            ...The scientists and academics said outlawing the cultivation of GM crops " risks constraining <a class="highlight">Scotland 's contribution to research</a> and  
            ...<br/>
            </blockquote>

        <p>You can highlight <strong>5 noun phrases</strong> at most. The maximum combined length of all highlights is <strong>20 words</strong>. Please note that for some cases, the bolded noun phrase is <strong>paraphrased</strong> in the document. You should also highlight the paraphrased phrases. <br/><br/>
        
        Also, please <strong>DO NOT</strong> highlight unrelated phrases. Here is a <strong>BAD</strong> example:
        <blockquote>
            <strong>Summary:</strong><br/>
            the queen has tweeted her thanks to people who sent her 90th birthday messages <strong>on social media</strong><br/>
            <em>(The noun phrase is: on social media)</em><br/><br/>
            <strong>Document:</strong><br/>
            ...and her official birthday held on <a class="highlight">a saturday in june</a> "...
            when she <a class="highlight">opened a new exhibition at the science museum in london</a>"
            ...
            the queen hosted the patron 's lunch ,<a class="highlight">a street party for some 10,000 people</a> along the mall which recognised her patronage
            ...
            <a class="highlight">queen elizabeth ii</a> at 90 find out more about queen <a class="highlight">elizabeth ii</a> on bbc iwonder
            ...<br/>
        </blockquote>


        <hr/>
        <p><strong>Step 2: Comprehension Question</strong></p>
        <p><strong>Please read the whole document before you do the highlighting</strong>. After you have submitted the evaluation, there will be one simple True/False question where we will verify whether you have read the document or not. Failure to answer the question correctly will invalidate the evaluation that the participant has done.</p>
        
        
        <hr/><p><strong>Step 3: Can you extract this bolded noun phrase present in the summary from the document?</strong></p>
        <p>Following the Comprehension Question, there is a track bar. Set a value from 1 to 3 by moving the indicator to show how clear this noun phrase in the summary is stated the document:  </p>
        
        <blockquote>
                <strong>3</strong>: The bolded noun phrase in the summary is explicitly described in the document. <br/>
                <strong>2</strong>: The bolded noun phrase is not explicitly mentioned in the document. However, it can be deduced from the document. <br/>
                <strong>1</strong>: The bolded noun phrase is not mentioned in the document at all. <br/>
        </blockquote>
        
        <p>There is a <strong>code</strong> that you have to copy and paste to Amazon platform at the end of the task. Only submissions that correctly answer the True/False question will be deemed as a valid HIT.</p>
        <hr/>
        <p>Thank you for reading this</p>

        <p>
            Do you consent for all the information that you have submitted to be used
            anonymously for research purposes only?
        </p>
        <!-- eslint-enable -->
    </div>
</template>

<script>
export default {
  name: 'LandingHighlightMTurk',
};
</script>
<style scoped lang="scss">
h3 {
    margin-top: 2rem;
    margin-bottom: 3rem;
}
.document {
  font-family: 'Lora', serif;
  font-size: 1.1rem;
  line-height: 1.5rem;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-bottom: 2rem;
}
.highlight{
  background-color: greenyellow;
}
</style>
