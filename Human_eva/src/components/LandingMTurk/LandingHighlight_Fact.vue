<template>
    <div class="document">
        <!-- eslint-disable -->
        <h3 align="center">
            Main Points Highlight
        </h3>
        <p>This task requires <strong>three</strong> steps:</p>
        <p><strong>Step 1: Highlighting Informative Phrases in a Document that Support the Bolded Part in a Given Summary the Best</strong></p>
        <p>In this step, you will be given two components <strong>a summary with one bolded main point</strong> and <strong>a document</strong>.</p>
        <p>You should:<br/></p>
            <blockquote>
            1) read the summary and identify the bolded main point <br/>
            2) read the document <br/>
            3) select sentences/phrases that support the bolded main point the best<br/>
            </blockquote>
        <p>An example of highlights:</p>
        
             <blockquote>
            <strong>Summary:</strong><br/>
            researchers have identified <strong>a gene that may put people at greater risk of strokes and heart attacks</strong>
            </blockquote>
            <em>(The main point you should identify is: a gene that may put people at greater risk of strokes and heart attacks)</em><br/><br/>
            <blockquote>
            <strong>Document:</strong><br/>
            ...<br/>
            Writing in PLOS ONE they say the <a class="highlight">gene fault may encourage the formation of blood clots - the ultimate cause of most heart attacks and strokes</a><br/>
            ...<br/>
            Around one in 10 people in the Caucasian population carries <a class="highlight">this variation of the gene , named PIA2</a><br/>
            ...<br/>
            They found <a class="highlight">individuals with PIA2 were more likely to have a stroke</a><br/>
            ...<br/>
            the scientists show <a class="highlight">PIA2 is also linked to an increased risk of heart attacks </a>in people under 45<br/>
            ...<br/>
            </blockquote>

        <p>You can highlight <strong>4 sentences/phrases</strong> at most. The maximum combined length of all highlights is <strong>50 words</strong>. 

        <hr/>
        <p><strong>Step 2: Comprehension Question</strong></p>
        <p><strong>Please read the whole document before you do the highlighting</strong>. After you have submitted the evaluation, there will be one simple True/False question where we will verify whether you have read the document or not. Failure to answer the question correctly will invalidate the evaluation that the participant has done.</p>
        
        
        <hr/><p><strong>Step 3: Can you extract this bolded main point present in the summary from the document?</strong></p>
        <p>Following the Comprehension Question, there is a track bar. Set a value from 1 to 3 by moving the indicator to show how clear this main point in the summary is stated the document:  </p>
        
        <blockquote>
                <strong>3</strong>: The bolded main point in the summary is explicitly described in the document. <br/>
                <strong>2</strong>: The bolded main point is not explicitly mentioned in the document. However, it can be deduced from the document. <br/>
                <strong>1</strong>: The bolded main point is not mentioned in the document at all. <br/>
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
